#!/usr/bin/env python3
"""
Test script for applying tensor decomposition to Qwen2.5-0.5B model using YAML configuration.
"""

import yaml
import re
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Tuple
from simple_tensor_decomposition import (
    apply_tensor_decomposition, 
    Decomposition,
    SimpleTRLinear,
    replace
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_yaml_rules(yaml_file: str, compression_factor: float = 1.0, topology: str = "btt") -> Dict[str, Tuple[Callable, Callable]]:
    """
    Parse YAML configuration file and convert to rules for tensor decomposition.
    
    Args:
        yaml_file: Path to YAML configuration file
        compression_factor: Compression factor to use (overrides YAML if needed)
        topology: Topology to use (overrides YAML if needed)
    
    Returns:
        Dictionary of rules for apply_tensor_decomposition
    """
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    rules = {}
    
    if 'rules' not in config:
        logger.warning("No 'rules' section found in YAML file")
        return rules
    
    for rule_name, rule_config in config['rules'].items():
        # Extract find pattern
        if 'find' in rule_config and 'pattern' in rule_config['find']:
            pattern = rule_config['find']['pattern']
        else:
            logger.warning(f"No pattern found for rule {rule_name}")
            continue
        
        # Extract replace configuration
        if 'general_replace' in rule_config:
            replace_config = rule_config['general_replace']
            
            # Extract solver method parameters
            solver_config = replace_config.get('solver_method', {})
            
            # Get parameters with defaults
            number_cores = solver_config.get('number_cores', 2)
            min_ranks = solver_config.get('min_ranks', 1)
            solver_method = "solve_low_ecf"  # Default for this YAML
            
            # Get shapes if specified
            in_shape = solver_config.get('in_shape')
            out_shape = solver_config.get('out_shape')
            
            # Use specified topology or default
            rule_topology = solver_config.get('topology', topology)
            
            # Create find function
            def make_find_func(pattern):
                def find_func(module, full_path):
                    if not isinstance(module, nn.Linear):
                        return False
                    return re.match(pattern, full_path) is not None
                return find_func
            
            # Create replace function
            def make_replace_func(comp_factor, topo, num_cores, min_r, solver_meth, in_sh, out_sh):
                def replace_func(module):
                    return replace(
                        module,
                        compression_factor=comp_factor,
                        topology=topo,
                        number_cores=num_cores,
                        min_ranks=min_r,
                        solver_method=solver_meth,
                        in_shape=in_sh,
                        out_shape=out_sh
                    )
                return replace_func
            
            find_func = make_find_func(pattern)
            replace_func = make_replace_func(
                compression_factor, rule_topology, number_cores, 
                min_ranks, solver_method, in_shape, out_shape
            )
            
            rules[rule_name] = (find_func, replace_func)
            logger.info(f"Created rule '{rule_name}' with pattern '{pattern}'")
    
    return rules


def load_qwen_model(model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu"):
    """
    Load Qwen2.5-0.5B model from Hugging Face.
    
    Args:
        model_name: Model name to load
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            device_map=device,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except ImportError:
        logger.error("transformers library not found. Please install it with: pip install transformers")
        return None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def analyze_model_layers(model: nn.Module):
    """Analyze model layers to understand structure."""
    logger.info("Analyzing model layers...")
    
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module.in_features, module.out_features))
    
    logger.info(f"Found {len(linear_layers)} linear layers:")
    for name, in_feat, out_feat in linear_layers[:10]:  # Show first 10
        logger.info(f"  {name}: {in_feat} -> {out_feat}")
    
    if len(linear_layers) > 10:
        logger.info(f"  ... and {len(linear_layers) - 10} more layers")
    
    return linear_layers


def test_qwen_tensor_decomposition():
    """Test tensor decomposition on Qwen2.5-0.5B model."""
    print("=== Testing Qwen2.5-0.5B Tensor Decomposition ===")
    
    # Load model
    model = load_qwen_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Analyze original model
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params:,}")
    
    # Analyze layers
    linear_layers = analyze_model_layers(model)
    
    # Parse YAML rules
    yaml_file = "ranks/wikitext2/Qwen2.5-0.5B/td_cfg/qwen_0.5B_td_btt.yaml"
    try:
        rules = parse_yaml_rules(yaml_file, compression_factor=1.0, topology="btt")
        print(f"Parsed {len(rules)} rules from YAML file")
        
        # Show rules
        for rule_name, (find_func, replace_func) in rules.items():
            print(f"  Rule: {rule_name}")
        
    except FileNotFoundError:
        print(f"YAML file not found: {yaml_file}")
        print("Creating sample rules for demonstration...")
        
        # Create sample rules based on the YAML structure
        rules = {}
        
        # Query projection rule
        def find_query(module, full_path):
            return isinstance(module, nn.Linear) and re.match(r"model\.layers\.\b([3-9]|[1][0-9]|[2][0-3])\b\.self_attn\.q_proj", full_path)
        
        def replace_query(module):
            return replace(module, compression_factor=1.0, topology="btt", number_cores=2, min_ranks=10, solver_method="solve_low_ecf", out_shape=[14, 64])
        
        rules["query"] = (find_query, replace_query)
        
        # Key projection rule
        def find_key(module, full_path):
            return isinstance(module, nn.Linear) and re.match(r"model\.layers\.\b([3-9]|[1][0-9]|[2][0-3])\b\.self_attn\.k_proj", full_path)
        
        def replace_key(module):
            return replace(module, compression_factor=1.0, topology="btt", number_cores=2, min_ranks=10, solver_method="solve_low_ecf", out_shape=[2, 64])
        
        rules["key"] = (find_key, replace_key)
        
        # Value projection rule
        def find_value(module, full_path):
            return isinstance(module, nn.Linear) and re.match(r"model\.layers\.\b([1][5-9]|[2][0-3])\b\.self_attn\.v_proj", full_path)
        
        def replace_value(module):
            return replace(module, compression_factor=1.0, topology="btt", number_cores=2, min_ranks=10, solver_method="solve_low_ecf", out_shape=[2, 64])
        
        rules["value"] = (find_value, replace_value)
        
        # Output projection rule
        def find_output(module, full_path):
            return isinstance(module, nn.Linear) and re.match(r"model\.layers\.\b([1][5-9]|[2][0-3])\b\.self_attn\.o_proj", full_path)
        
        def replace_output(module):
            return replace(module, compression_factor=1.0, topology="btt", number_cores=2, min_ranks=10, solver_method="solve_low_ecf", in_shape=[14, 64])
        
        rules["causal_attention_projection"] = (find_output, replace_output)
        
        # MLP rules
        def find_mlp_up(module, full_path):
            return isinstance(module, nn.Linear) and re.match(r"model\.layers\.\b([7-9]|[1][0-9]|[2][0-3])\b\.mlp\.up_proj", full_path)
        
        def replace_mlp_up(module):
            return replace(module, compression_factor=1.0, topology="btt", number_cores=2, min_ranks=10, solver_method="solve_low_ecf")
        
        rules["mlp_up_proj"] = (find_mlp_up, replace_mlp_up)
        
        def find_mlp_down(module, full_path):
            return isinstance(module, nn.Linear) and re.match(r"model\.layers\.\b([1][5-9]|[2][0-3])\b\.mlp\.down_proj", full_path)
        
        def replace_mlp_down(module):
            return replace(module, compression_factor=1.0, topology="btt", number_cores=2, min_ranks=10, solver_method="solve_low_ecf")
        
        rules["mlp_down_proj"] = (find_mlp_down, replace_mlp_down)
        
        def find_mlp_gate(module, full_path):
            return isinstance(module, nn.Linear) and re.match(r"model\.layers\.\b([7-9]|1[0-9]|2[0-3])\b\.mlp\.gate_proj", full_path)
        
        def replace_mlp_gate(module):
            return replace(module, compression_factor=1.0, topology="btt", number_cores=2, min_ranks=10, solver_method="solve_low_ecf")
        
        rules["mlp_gate_proj"] = (find_mlp_gate, replace_mlp_gate)
    
    # Apply tensor decomposition
    print("\nApplying tensor decomposition...")
    apply_tensor_decomposition(model, rules=rules)
    
    # Analyze compressed model
    compressed_params = sum(p.numel() for p in model.parameters())
    compression_ratio = original_params / compressed_params
    
    print(f"\nCompression results:")
    print(f"  Original parameters: {original_params:,}")
    print(f"  Compressed parameters: {compressed_params:,}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Parameter reduction: {((original_params - compressed_params) / original_params * 100):.1f}%")
    
    # Check which layers were replaced
    td_layers = []
    for name, module in model.named_modules():
        if isinstance(module, SimpleTRLinear):
            td_layers.append(name)
    
    print(f"\nReplaced {len(td_layers)} layers with tensor decomposition:")
    for layer_name in td_layers[:10]:  # Show first 10
        print(f"  {layer_name}")
    
    if len(td_layers) > 10:
        print(f"  ... and {len(td_layers) - 10} more layers")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        # Create a simple input
        input_ids = torch.randint(0, 1000, (1, 10))  # Batch size 1, sequence length 10
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"Forward pass successful!")
        print(f"Output shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


def test_simple_model_with_yaml():
    """Test with a simple model to verify YAML parsing works."""
    print("\n=== Testing Simple Model with YAML Rules ===")
    
    # Create a simple model that mimics Qwen structure
    class SimpleQwenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList()
            
            # Create some layers that match the YAML patterns
            for i in range(24):  # 24 layers like Qwen
                layer = nn.Module()
                
                # Self attention
                layer.self_attn = nn.Module()
                layer.self_attn.q_proj = nn.Linear(896, 896)  # 896 = 14 * 64
                layer.self_attn.k_proj = nn.Linear(896, 128)  # 128 = 2 * 64
                layer.self_attn.v_proj = nn.Linear(896, 128)  # 128 = 2 * 64
                layer.self_attn.o_proj = nn.Linear(128, 896)  # 128 = 14 * 64
                
                # MLP
                layer.mlp = nn.Module()
                layer.mlp.up_proj = nn.Linear(896, 1792)
                layer.mlp.down_proj = nn.Linear(1792, 896)
                layer.mlp.gate_proj = nn.Linear(896, 1792)
                
                self.model.layers.append(layer)
        
        def forward(self, x):
            for layer in self.model.layers:
                # Simplified forward pass
                x = layer.self_attn.q_proj(x)
                x = layer.self_attn.o_proj(x)
                x = layer.mlp.up_proj(x)
                x = layer.mlp.down_proj(x)
            return x
    
    model = SimpleQwenModel()
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Simple model parameters: {original_params:,}")
    
    # Create rules similar to the YAML
    rules = {}
    
    def find_query(module, full_path):
        return isinstance(module, nn.Linear) and "q_proj" in full_path
    
    def replace_query(module):
        return replace(module, compression_factor=1.0, topology="btt", number_cores=2, min_ranks=10, solver_method="solve_low_ecf", out_shape=[14, 64])
    
    rules["query"] = (find_query, replace_query)
    
    # Apply tensor decomposition
    apply_tensor_decomposition(model, rules=rules)
    
    compressed_params = sum(p.numel() for p in model.parameters())
    print(f"Compressed parameters: {compressed_params:,}")
    print(f"Compression ratio: {original_params / compressed_params:.2f}x")
    
    # Test forward pass
    x = torch.randn(1, 896)
    with torch.no_grad():
        output = model(x)
    print(f"Forward pass successful! Output shape: {output.shape}")


if __name__ == "__main__":
    print("Testing Qwen2.5-0.5B Tensor Decomposition with YAML Configuration")
    print("=" * 70)
    
    try:
        # Test with simple model first
        test_simple_model_with_yaml()
        
        # Test with actual Qwen model
        test_qwen_tensor_decomposition()
        
        print("\n" + "=" * 70)
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 