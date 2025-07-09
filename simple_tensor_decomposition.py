"""
Self-contained Tensor Decomposition implementation for PyTorch models.

This module provides a simplified version of the tensor decomposition functionality
from the percipio2 package, allowing you to apply tensor decomposition to any PyTorch model.
"""

import re
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Decomposition:
    """Configuration for tensor decomposition layers."""
    in_shape: List[int]
    out_shape: List[int]
    ranks: List[int]
    topology: str = "mpo"
    
    @classmethod
    def solve_mpo(cls, in_shape: List[int], out_shape: List[int], compression_factor: float) -> 'Decomposition':
        """Solve for MPO decomposition configuration."""
        assert len(in_shape) == len(out_shape), "For MPO topology, input and output must have same number of dimensions"
        
        # Simple rank calculation based on compression factor
        total_params = sum(i * o for i, o in zip(in_shape, out_shape))
        target_params = total_params / compression_factor
        
        # Calculate ranks (simplified approach)
        n_cores = len(in_shape)
        if n_cores == 1:
            ranks = [1]
        else:
            # Simple rank calculation - can be improved
            base_rank = max(1, int((target_params / n_cores) ** 0.5))
            ranks = [base_rank] * n_cores
            ranks[0] = 1  # First rank is typically 1 for MPO
        
        return cls(in_shape=in_shape, out_shape=out_shape, ranks=ranks, topology="mpo")
    
    @classmethod
    def solve_btt(cls, in_shape: List[int], out_shape: List[int], compression_factor: float) -> 'Decomposition':
        """Solve for BTT decomposition configuration."""
        assert len(in_shape) == len(out_shape), "For BTT topology, input and output must have same number of dimensions"
        
        # Simple rank calculation for BTT
        total_params = sum(i * o for i, o in zip(in_shape, out_shape))
        target_params = total_params / compression_factor
        
        n_cores = len(in_shape)
        if n_cores == 1:
            ranks = [1]
        else:
            # BTT typically uses different rank structure
            base_rank = max(1, int((target_params / n_cores) ** 0.5))
            ranks = [base_rank] * n_cores
        
        return cls(in_shape=in_shape, out_shape=out_shape, ranks=ranks, topology="btt")
    
    @classmethod
    def solve_low_ecf(cls, in_channels: int, out_channels: int, number_cores: int, 
                     compression_factor: float, min_ranks: int, topology: str = "mpo",
                     in_shape: Optional[List[int]] = None, out_shape: Optional[List[int]] = None) -> 'Decomposition':
        """Solve for low effective compression factor configuration."""
        # Determine shapes
        if in_shape is None:
            # Try to factorize in_channels into number_cores factors
            factors = cls._find_factors(in_channels, number_cores)
            in_shape = factors
        else:
            assert len(in_shape) == number_cores, f"in_shape length {len(in_shape)} != number_cores {number_cores}"
        
        if out_shape is None:
            # Try to factorize out_channels into number_cores factors
            factors = cls._find_factors(out_channels, number_cores)
            out_shape = factors
        else:
            assert len(out_shape) == number_cores, f"out_shape length {len(out_shape)} != number_cores {number_cores}"
        
        # Calculate ranks to achieve compression_factor
        total_params = sum(i * o for i, o in zip(in_shape, out_shape))
        target_params = total_params / compression_factor
        
        # Calculate ranks ensuring minimum rank constraint
        if number_cores == 1:
            ranks = [max(min_ranks, 1)]
        else:
            # Simple rank calculation with minimum constraint
            base_rank = max(min_ranks, int((target_params / number_cores) ** 0.5))
            ranks = [base_rank] * number_cores
        
        return cls(in_shape=in_shape, out_shape=out_shape, ranks=ranks, topology=topology)
    
    @staticmethod
    def _find_factors(n: int, num_factors: int) -> List[int]:
        """Find factors of n that can be split into num_factors parts."""
        if num_factors == 1:
            return [n]
        
        # Simple factorization - find factors that multiply to n
        factors = []
        remaining = n
        
        for i in range(num_factors - 1):
            # Find a factor that divides remaining
            factor = int(remaining ** (1 / (num_factors - i)))
            while remaining % factor != 0 and factor > 1:
                factor -= 1
            factors.append(factor)
            remaining //= factor
        
        factors.append(remaining)
        return factors


class SimpleTRLinear(nn.Module):
    """Simplified Tensor Ring Linear layer."""
    
    def __init__(self, decomposition: Decomposition, bias: bool = False):
        super().__init__()
        self.decomposition = decomposition
        self.bias = bias
        
        # Create core tensors
        self.cores = nn.ParameterList()
        self._setup_cores()
        
        if bias:
            out_size = self._get_output_size()
            self.bias_params = nn.Parameter(torch.zeros(out_size))
    
    def _setup_cores(self):
        """Setup tensor cores based on topology."""
        if self.decomposition.topology == "mpo":
            self._setup_mpo_cores()
        elif self.decomposition.topology == "btt":
            self._setup_btt_cores()
        else:
            raise ValueError(f"Topology {self.decomposition.topology} not supported in simplified version")
    
    def _setup_mpo_cores(self):
        """Setup MPO cores."""
        in_shape = self.decomposition.in_shape
        out_shape = self.decomposition.out_shape
        ranks = self.decomposition.ranks
        
        n_cores = len(in_shape)
        
        for i in range(n_cores):
            if i == 0:
                # First core: [in_shape[0], out_shape[0], ranks[0]]
                core_shape = (in_shape[0], out_shape[0], ranks[0])
            elif i == n_cores - 1:
                # Last core: [ranks[-1], in_shape[-1], out_shape[-1]]
                core_shape = (ranks[-1], in_shape[-1], out_shape[-1])
            else:
                # Middle cores: [ranks[i-1], in_shape[i], out_shape[i], ranks[i]]
                core_shape = (ranks[i-1], in_shape[i], out_shape[i], ranks[i])
            
            core = nn.Parameter(torch.randn(*core_shape) * 0.1)
            self.cores.append(core)
    
    def _get_output_size(self) -> int:
        """Calculate total output size."""
        return self.decomposition.out_shape[0] if len(self.decomposition.out_shape) == 1 else sum(self.decomposition.out_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through tensor network."""
        if self.decomposition.topology == "mpo":
            return self._forward_mpo(x)
        elif self.decomposition.topology == "btt":
            return self._forward_btt(x)
        else:
            raise ValueError(f"Topology {self.decomposition.topology} not supported")
    
    def _forward_mpo(self, x: torch.Tensor) -> torch.Tensor:
        """MPO forward pass (simplified implementation)."""
        # For single-core MPO, we can use a simpler approach
        if len(self.cores) == 1:
            # Single core: direct matrix multiplication
            core = self.cores[0]
            # Reshape core to 2D matrix
            core_2d = core.view(core.shape[0], -1)
            result = torch.matmul(x, core_2d)
        else:
            # Multi-core MPO (simplified)
            batch_size = x.shape[0]
            result = x
            
            for i, core in enumerate(self.cores):
                if i == 0:
                    # First core: [in_shape[0], out_shape[0], ranks[0]]
                    core_reshaped = core.view(core.shape[0], -1)
                    result = torch.matmul(result, core_reshaped)
                elif i == len(self.cores) - 1:
                    # Last core: [ranks[-1], in_shape[-1], out_shape[-1]]
                    core_reshaped = core.view(core.shape[0], -1)
                    result = torch.matmul(result, core_reshaped)
                else:
                    # Middle cores: [ranks[i-1], in_shape[i], out_shape[i], ranks[i]]
                    core_reshaped = core.view(core.shape[0], -1)
                    result = torch.matmul(result, core_reshaped)
        
        if self.bias:
            result = result + self.bias_params
        
        return result
    
    def _forward_btt(self, x: torch.Tensor) -> torch.Tensor:
        """BTT forward pass (simplified implementation)."""
        # For single-core BTT, we can use a simpler approach
        if len(self.cores) == 1:
            # Single core: direct matrix multiplication
            core = self.cores[0]
            # Reshape core to 2D matrix
            core_2d = core.view(core.shape[0], -1)
            result = torch.matmul(x, core_2d)
        else:
            # Multi-core BTT (simplified)
            result = x
            
            for i, core in enumerate(self.cores):
                if i == 0:
                    # First core: [in_shape[0], out_shape[0], ranks[0]]
                    core_reshaped = core.view(core.shape[0], -1)
                    result = torch.matmul(result, core_reshaped)
                elif i == len(self.cores) - 1:
                    # Last core: [ranks[-1], in_shape[-1], out_shape[-1]]
                    core_reshaped = core.view(core.shape[0], -1)
                    result = torch.matmul(result, core_reshaped)
                else:
                    # Middle cores: [ranks[i-1], in_shape[i], out_shape[i], ranks[i]]
                    core_reshaped = core.view(core.shape[0], -1)
                    result = torch.matmul(result, core_reshaped)
        
        if self.bias:
            result = result + self.bias_params
        
        return result
    
    def _setup_btt_cores(self):
        """Setup BTT cores."""
        in_shape = self.decomposition.in_shape
        out_shape = self.decomposition.out_shape
        ranks = self.decomposition.ranks
        
        n_cores = len(in_shape)
        
        for i in range(n_cores):
            if i == 0:
                # First core: [in_shape[0], out_shape[0], ranks[0]]
                core_shape = (in_shape[0], out_shape[0], ranks[0])
            elif i == n_cores - 1:
                # Last core: [ranks[-1], in_shape[-1], out_shape[-1]]
                core_shape = (ranks[-1], in_shape[-1], out_shape[-1])
            else:
                # Middle cores: [ranks[i-1], in_shape[i], out_shape[i], ranks[i]]
                core_shape = (ranks[i-1], in_shape[i], out_shape[i], ranks[i])
            
            core = nn.Parameter(torch.randn(*core_shape) * 0.1)
            self.cores.append(core)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "in_shape": self.decomposition.in_shape,
            "out_shape": self.decomposition.out_shape,
            "ranks": self.decomposition.ranks,
            "bias": self.bias,
            "topology": self.decomposition.topology,
        }
    
    @classmethod
    def create_from_decomposition(cls, decomposition: Decomposition, bias: bool = False) -> 'SimpleTRLinear':
        """Create TRLinear from decomposition config."""
        return cls(decomposition=decomposition, bias=bias)


def find(module: nn.Module, full_path: str, pattern: str) -> bool:
    """Check if module is a candidate for replacement."""
    if not isinstance(module, nn.Linear):
        return False
    return re.match(pattern, full_path) is not None


def replace(
    module: nn.Linear,
    compression_factor: float = 2.0,
    topology: str = "mpo",
    in_shape: Optional[List[int]] = None,
    out_shape: Optional[List[int]] = None,
    ranks: Optional[List[int]] = None,
    number_cores: int = 2,
    min_ranks: int = 1,
    solver_method: str = "solve_mpo",
) -> SimpleTRLinear:
    """Replace a linear module with a tensor decomposition module."""
    assert isinstance(module, nn.Linear), f"Cannot replace module of type {type(module)}"
    
    # Determine shapes
    if in_shape is None:
        in_shape = [module.in_features]
    if out_shape is None:
        out_shape = [module.out_features]
    
    # Create decomposition config based on solver method
    if solver_method == "solve_low_ecf":
        decomposition = Decomposition.solve_low_ecf(
            in_channels=module.in_features,
            out_channels=module.out_features,
            number_cores=number_cores,
            compression_factor=compression_factor,
            min_ranks=min_ranks,
            topology=topology,
            in_shape=in_shape,
            out_shape=out_shape
        )
    elif solver_method == "solve_btt":
        decomposition = Decomposition.solve_btt(in_shape, out_shape, compression_factor)
    elif solver_method == "solve_mpo":
        decomposition = Decomposition.solve_mpo(in_shape, out_shape, compression_factor)
    else:
        # Use provided ranks or default
        if ranks is None:
            decomposition = Decomposition.solve_mpo(in_shape, out_shape, compression_factor)
        else:
            decomposition = Decomposition(in_shape=in_shape, out_shape=out_shape, ranks=ranks, topology=topology)
    
    # Create TRLinear module
    td_module = SimpleTRLinear.create_from_decomposition(
        decomposition,
        bias=module.bias is not None
    )
    
    # Copy weights if possible (simplified - in practice you'd decompose the weights)
    device = module.weight.device
    dtype = module.weight.dtype
    td_module = td_module.to(device=device, dtype=dtype)
    
    logger.debug(f"Replaced module {module} with TD module {td_module}")
    
    return td_module


def find_replace(
    module: nn.Module,
    rules: Dict[str, Tuple[Callable, Callable]],
    path_prefix: Optional[str] = None
) -> None:
    """Recursively replace modules according to find/replace rules."""
    for k, submodule in module.named_children():
        full_path = ".".join([path_prefix, k]) if path_prefix is not None else k
        
        # Apply rules in reverse order (most specific first)
        for rule_name, (find_func, replace_func) in reversed(list(rules.items())):
            if find_func(submodule, full_path):
                logger.debug(f'Replacing module "{full_path}" according to rule: {rule_name}')
                new_submodule = replace_func(submodule)
                setattr(module, k, new_submodule)
                break  # Stop after first match
        
        # Recursively apply to submodules
        find_replace(submodule, rules, full_path)


def apply_tensor_decomposition(
    model: nn.Module,
    rules: Optional[Dict[str, Tuple[Callable, Callable]]] = None,
    compression_factor: float = 2.0,
    topology: str = "mpo",
    pattern: str = ".*"
) -> None:
    """
    Apply tensor decomposition to a PyTorch model.
    
    Args:
        model: The PyTorch model to apply tensor decomposition to
        rules: Dictionary mapping rule names to (find, replace) function pairs.
               If None, uses default rules based on pattern and compression_factor.
        compression_factor: Compression factor for tensor decomposition (default: 2.0)
        topology: Tensor network topology ("mpo", "mps", "btt") (default: "mpo")
        pattern: Regex pattern to match module names (default: ".*" for all Linear layers)
    
    Example:
        # Apply TD to all linear layers with 2x compression
        apply_tensor_decomposition(model, compression_factor=2.0)
        
        # Apply TD only to attention layers
        apply_tensor_decomposition(model, pattern=r".*attention.*", compression_factor=3.0)
        
        # Use custom rules
        custom_rules = {
            "attention": (
                lambda m, p: "attention" in p and isinstance(m, nn.Linear),
                lambda m: replace(m, compression_factor=3.0)
            )
        }
        apply_tensor_decomposition(model, rules=custom_rules)
    """
    if rules is None:
        # Create default rules
        def find_func(module, full_path):
            return find(module, full_path, pattern)
        
        def replace_func(module):
            return replace(module, compression_factor=compression_factor, topology=topology)
        
        rules = {"default": (find_func, replace_func)}
    
    logger.info(f"Applying tensor decomposition to model with {len(rules)} rules")
    find_replace(model, rules)
    logger.info("Tensor decomposition applied successfully")


# Example usage and utility functions
def create_attention_rules(compression_factor: float = 2.0) -> Dict[str, Tuple[Callable, Callable]]:
    """Create rules for attention layer decomposition."""
    def find_attention(module, full_path):
        return (isinstance(module, nn.Linear) and 
                any(key in full_path.lower() for key in ['attention', 'attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj']))
    
    def replace_attention(module):
        return replace(module, compression_factor=compression_factor)
    
    return {"attention": (find_attention, replace_attention)}


def create_mlp_rules(compression_factor: float = 2.0) -> Dict[str, Tuple[Callable, Callable]]:
    """Create rules for MLP layer decomposition."""
    def find_mlp(module, full_path):
        return (isinstance(module, nn.Linear) and 
                any(key in full_path.lower() for key in ['mlp', 'ffn', 'feed_forward', 'fc']))
    
    def replace_mlp(module):
        return replace(module, compression_factor=compression_factor)
    
    return {"mlp": (find_mlp, replace_mlp)}


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 256)
            self.linear2 = nn.Linear(256, 128)
            self.attention = nn.Linear(128, 128)
            self.output = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.attention(x)
            x = self.output(x)
            return x
    
    # Create model
    model = SimpleModel()
    print("Original model:")
    print(model)
    
    # Apply tensor decomposition
    apply_tensor_decomposition(model, compression_factor=2.0)
    
    print("\nModel after tensor decomposition:")
    print(model) 