# 🚀 FLAT-LLM

This is the official PyTorch implementation of **FLAT-LLM** **F**ine-grained **L**ow-rank **A**ctivation Space **T**ransformation for Large Language Model Compression [arxiv](https://arxiv.org/pdf/2505.23966)

---

## 📦 Environment Setup

Installation instructions can be found in [INSTALL.md](INSTALL.md).

---

## 🛠️ Run the Code

All scripts for reproducing our main results (Table 1) are available in the [`scripts`](scripts) directory.

### 🔍 Importance-Preserving Rank Selection (IPRS)

1. Run `llama_bi.sh` to compute decoder-wise importance scores.
2. Run `compute_rank.py` to 
    - Allocate ranks with our IPRS algorithm according to the importance scores.
    - Compute the compression ratio for V,O,MLP layers (Q,K are not pruned) according to required total compresstion ratio.

### ✂️ FLAT-LLM Pruning

Run one of the following scripts to prune and evaluate the corresponding model:
- `llama_7b.sh` # use 1 A100 40GB
- `llama_13b.sh` # use 1 A100 40GB
- `llama_70b.sh` # use 4 A100 40GB
- `mistral.sh` # use 1 A100 40GB

These reproduce the perplexity results reported in Table 1 of the paper when using wikitext2 for calibration.

---

## 🔧 Command-Line Arguments

### 📦 Model and Dataset
- `--model`: Name or path of the LLM to prune. Choices: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-2-70b-hf`, `mistralai/Mistral-7B-v0.1`
- `--dataset`: Calibration dataset. Choices: `wikitext2`, `c4`, `alpaca`.
- `--cache_dir`: Directory to cache model weights.

### ⚙️ Pruning Configuration
- `--prune_method`: Pruning stage. Options:
  - `bi`: Rank allocation via importance scores.
  - `flatllm`: Final pruning using head-wise PCA.
- `--sparsity_ratio`: Target sparsity level (as an integer percentage).
- `--tol`: Tolerance threshold on cumulative eigenvalues. Default: `0.96`. (this hyper-para is only for monitoring the calibration, not used in the algorithm)
- `--bi_score`: Path to save/load the importance scores/allocated ranks.
- `--seed`: Random seed for reproducibility.
- `--nsamples`: Number of calibration samples.
- `--save`: Path to save logs.
- `--save_model`: Path to save the pruned model.

---

## 📊 Evaluation

### 🧠 Zero-Shot Evaluation

We evaluate zero-shot downstream task performance using the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness). Please use the modified code for zero-shot/few-shot evaluation in [lm_eval](https://github.com/TTTTTTris/lm_eval) repo.

### ⚡ Inference Speedup

To benchmark inference speedup, we build upon the evaluation framework from [SliceGPT](https://github.com/microsoft/TransformerCompression).

---

## 📄 License

This project is licensed under the MIT License. 

# Self-Contained Tensor Decomposition

This is a simplified, self-contained implementation of tensor decomposition for PyTorch models, extracted from the `percipio2` package. It allows you to apply tensor decomposition to any PyTorch model without requiring the full percipio2 dependency.

## What is Tensor Decomposition?

Tensor decomposition is a technique for compressing neural network layers by representing weight matrices as tensor networks. This can significantly reduce the number of parameters while maintaining model performance.

## Features

- **Self-contained**: No external dependencies beyond PyTorch
- **Easy to use**: Simple API for applying tensor decomposition to any PyTorch model
- **Flexible**: Support for custom rules and selective layer replacement
- **Multiple topologies**: Support for MPO (Matrix Product Operator) topology
- **Configurable compression**: Adjustable compression factors

## Installation

No installation required! Just copy the `simple_tensor_decomposition.py` file to your project.

Requirements:
- PyTorch (>= 1.8.0)
- Python (>= 3.7)

## Quick Start

```python
import torch.nn as nn
from simple_tensor_decomposition import apply_tensor_decomposition

# Create your model
class MyModel(nn.Module):
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

# Create model
model = MyModel()

# Apply tensor decomposition with 2x compression
apply_tensor_decomposition(model, compression_factor=2.0)

# Your model is now compressed!
```

## Usage Examples

### 1. Basic Usage

Apply tensor decomposition to all linear layers:

```python
apply_tensor_decomposition(model, compression_factor=2.0)
```

### 2. Selective Decomposition

Apply tensor decomposition only to specific layers using regex patterns:

```python
# Only attention layers
apply_tensor_decomposition(model, pattern=r".*attention.*", compression_factor=3.0)

# Only projection layers
apply_tensor_decomposition(model, pattern=r".*_proj", compression_factor=2.5)
```

### 3. Custom Rules

Create custom rules for more control:

```python
from simple_tensor_decomposition import create_attention_rules, create_mlp_rules

# Create rules for different layer types
attention_rules = create_attention_rules(compression_factor=2.0)
mlp_rules = create_mlp_rules(compression_factor=3.0)

# Combine rules
all_rules = {**attention_rules, **mlp_rules}

# Apply custom rules
apply_tensor_decomposition(model, rules=all_rules)
```

### 4. Manual Configuration

For advanced users, you can manually configure decomposition:

```python
from simple_tensor_decomposition import Decomposition, SimpleTRLinear

# Create custom decomposition config
decomp = Decomposition(
    in_shape=[16, 16],   # 256 = 16 * 16
    out_shape=[16, 8],   # 128 = 16 * 8
    ranks=[1, 4],        # MPO ranks
    topology="mpo"
)

# Create TRLinear layer
td_layer = SimpleTRLinear.create_from_decomposition(decomp, bias=True)
```

## API Reference

### `apply_tensor_decomposition(model, rules=None, compression_factor=2.0, topology="mpo", pattern=".*")`

Apply tensor decomposition to a PyTorch model.

**Parameters:**
- `model` (nn.Module): The PyTorch model to compress
- `rules` (dict, optional): Custom rules dictionary. If None, uses default rules
- `compression_factor` (float): Target compression factor (default: 2.0)
- `topology` (str): Tensor network topology ("mpo", "mps", "btt") (default: "mpo")
- `pattern` (str): Regex pattern for layer matching (default: ".*")

### `create_attention_rules(compression_factor=2.0)`

Create predefined rules for attention layers.

### `create_mlp_rules(compression_factor=2.0)`

Create predefined rules for MLP/feed-forward layers.

### `Decomposition`

Configuration class for tensor decomposition:

```python
@dataclass
class Decomposition:
    in_shape: List[int]      # Input tensor shape
    out_shape: List[int]     # Output tensor shape
    ranks: List[int]         # Tensor network ranks
    topology: str = "mpo"    # Network topology
```

### `SimpleTRLinear`

Tensor Ring Linear layer that replaces standard linear layers.

## Testing

Run the test script to see examples in action:

```bash
python test_tensor_decomposition.py
```

## Limitations

This simplified version has some limitations compared to the full percipio2 implementation:

1. **Limited topologies**: Only MPO topology is fully implemented
2. **Simplified decomposition**: Uses basic rank calculation
3. **No weight transfer**: Doesn't transfer weights from original layers
4. **Basic tensor contraction**: Simplified forward pass implementation

## Advanced Usage

### Custom Find/Replace Rules

You can create completely custom rules:

```python
def find_my_layers(module, full_path):
    return isinstance(module, nn.Linear) and "my_special_layer" in full_path

def replace_my_layers(module):
    return replace(module, compression_factor=4.0, topology="mpo")

custom_rules = {
    "my_rule": (find_my_layers, replace_my_layers)
}

apply_tensor_decomposition(model, rules=custom_rules)
```

### Checking Compression Effectiveness

```python
# Count parameters before and after
original_params = sum(p.numel() for p in model.parameters())
apply_tensor_decomposition(model, compression_factor=2.0)
compressed_params = sum(p.numel() for p in model.parameters())

print(f"Compression ratio: {original_params / compressed_params:.2f}x")
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you have PyTorch installed
2. **Pattern not matching**: Check your regex pattern with `print(model)` to see layer names
3. **Compression not working**: Verify that your model has `nn.Linear` layers

### Debug Mode

Enable debug logging to see which layers are being replaced:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

This is a simplified version for educational and practical use. For the full implementation with advanced features, see the original percipio2 package.

## License

This code is provided as-is for educational purposes. Please refer to the original percipio2 package for licensing information. 
