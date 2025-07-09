#!/usr/bin/env python3
"""
Test script for the self-contained tensor decomposition implementation.
"""

import torch
import torch.nn as nn
from simple_tensor_decomposition import (
    apply_tensor_decomposition, 
    create_attention_rules, 
    create_mlp_rules,
    Decomposition,
    SimpleTRLinear
)

def test_basic_usage():
    """Test basic tensor decomposition usage."""
    print("=== Testing Basic Usage ===")
    
    # Create a simple model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 256)
            self.linear2 = nn.Linear(256, 128)
            self.output = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.output(x)
            return x
    
    model = TestModel()
    print("Original model:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Apply tensor decomposition to all linear layers
    apply_tensor_decomposition(model, compression_factor=2.0)
    
    print("\nModel after tensor decomposition:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    x = torch.randn(2, 512)
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")


def test_selective_decomposition():
    """Test selective tensor decomposition using patterns."""
    print("\n=== Testing Selective Decomposition ===")
    
    class AttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(100, 512)
            self.q_proj = nn.Linear(512, 512)
            self.k_proj = nn.Linear(512, 512)
            self.v_proj = nn.Linear(512, 512)
            self.o_proj = nn.Linear(512, 512)
            self.mlp1 = nn.Linear(512, 1024)
            self.mlp2 = nn.Linear(1024, 512)
            self.output = nn.Linear(512, 100)
        
        def forward(self, x):
            x = self.embedding(x)
            # Attention (simplified)
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            attn_out = self.o_proj(q)  # Simplified attention
            # MLP
            x = self.mlp1(attn_out)
            x = self.mlp2(x)
            x = self.output(x)
            return x
    
    model = AttentionModel()
    print("Original model:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Apply TD only to attention layers
    apply_tensor_decomposition(model, pattern=r".*_proj", compression_factor=3.0)
    
    print("\nModel after selective tensor decomposition:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Check which layers were replaced
    for name, module in model.named_modules():
        if isinstance(module, SimpleTRLinear):
            print(f"Replaced: {name} -> {type(module).__name__}")


def test_custom_rules():
    """Test custom rules for tensor decomposition."""
    print("\n=== Testing Custom Rules ===")
    
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_block = nn.Sequential(
                nn.Linear(256, 256),
                nn.Linear(256, 256)
            )
            self.feed_forward = nn.Sequential(
                nn.Linear(256, 512),
                nn.Linear(512, 256)
            )
            self.classifier = nn.Linear(256, 10)
        
        def forward(self, x):
            x = self.attention_block(x)
            x = self.feed_forward(x)
            x = self.classifier(x)
            return x
    
    model = ComplexModel()
    print("Original model:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create custom rules
    attention_rules = create_attention_rules(compression_factor=2.0)
    mlp_rules = create_mlp_rules(compression_factor=3.0)
    
    # Combine rules
    all_rules = {**attention_rules, **mlp_rules}
    
    # Apply custom rules
    apply_tensor_decomposition(model, rules=all_rules)
    
    print("\nModel after custom rules tensor decomposition:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Check which layers were replaced
    for name, module in model.named_modules():
        if isinstance(module, SimpleTRLinear):
            print(f"Replaced: {name} -> {type(module).__name__}")


def test_decomposition_config():
    """Test manual decomposition configuration."""
    print("\n=== Testing Manual Decomposition Config ===")
    
    # Create a simple linear layer
    linear = nn.Linear(100, 200)
    
    # Create custom decomposition config
    decomp = Decomposition(
        in_shape=[10, 10],  # 100 = 10 * 10
        out_shape=[10, 20], # 200 = 10 * 20
        ranks=[1, 5],       # MPO ranks
        topology="mpo"
    )
    
    # Create TRLinear manually
    td_layer = SimpleTRLinear.create_from_decomposition(decomp, bias=True)
    
    print(f"Original linear layer parameters: {sum(p.numel() for p in linear.parameters())}")
    print(f"TRLinear layer parameters: {sum(p.numel() for p in td_layer.parameters())}")
    
    # Test forward pass
    x = torch.randn(2, 100)
    with torch.no_grad():
        linear_out = linear(x)
        td_out = td_layer(x)
    
    print(f"Linear output shape: {linear_out.shape}")
    print(f"TRLinear output shape: {td_out.shape}")


def test_compression_effectiveness():
    """Test the effectiveness of compression."""
    print("\n=== Testing Compression Effectiveness ===")
    
    # Create a large linear layer
    large_linear = nn.Linear(1024, 1024)
    original_params = sum(p.numel() for p in large_linear.parameters())
    print(f"Original linear layer parameters: {original_params}")
    
    # Apply different compression factors
    for cf in [2.0, 4.0, 8.0]:
        # Create a copy for testing
        test_linear = nn.Linear(1024, 1024)
        
        # Create decomposition
        decomp = Decomposition.solve_mpo([32, 32], [32, 32], cf)  # 1024 = 32 * 32
        td_layer = SimpleTRLinear.create_from_decomposition(decomp, bias=True)
        
        td_params = sum(p.numel() for p in td_layer.parameters())
        compression_ratio = original_params / td_params
        
        print(f"Compression factor {cf}: {td_params} parameters, actual compression: {compression_ratio:.2f}x")


if __name__ == "__main__":
    print("Testing Self-Contained Tensor Decomposition")
    print("=" * 50)
    
    try:
        test_basic_usage()
        test_selective_decomposition()
        test_custom_rules()
        test_decomposition_config()
        test_compression_effectiveness()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 