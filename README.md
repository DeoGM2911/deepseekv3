# DeepSeek V3.2 Educational Implementation

This repository is an educational implementation of the **DeepSeek V3.2** architecture, designed to help understand the core mechanisms behind this state-of-the-art language model. This implementation focuses on clarity and learning rather than production-level optimization.

> **Note**: This is a simplified educational replica created for learning purposes. For production use, please refer to the official DeepSeek implementation.

## Overview

DeepSeek V3.2 introduces two groundbreaking mechanisms:
- **Multi-head Latent Attention (MLA)**: An efficient attention mechanism that reduces KV cache memory requirements
- **DeepSeek Sparse Attention (DSA)**: A sparse attention mechanism built on top of MLA that selectively attends to relevant tokens
- **Group Relative Policy Optimization (GRPO)**: A reinforcement learning framework for fine-tuning the model. This is introduced
with DeepSeek V3 and R1, but it is still used for fine-tuning DeepSeek V3.2. 

## Project Structure

```
deepseekv3/
├── model.py                    # Main model architecture
├── inference.py                # Inference utilities and generation
├── test_model.py              # Model testing script
├── requirements.txt           # Python dependencies
├── deepseek_utils/            # Core components
│   ├── __init__.py
│   ├── attention.py           # MLA and DSA implementation
│   ├── decoder.py             # Decoder block
│   ├── moe.py                 # Mixture of Experts
│   ├── utils.py               # Utility functions (RoPE, etc.)
│   ├── test_attention.py      # Attention mechanism tests
│   ├── test_decoder.py        # Decoder block tests
│   └── test_moe.py            # MoE tests
├── train/                     # Training utilities
│   ├── __init__.py
│   ├── trainer.py             # Training loop and configuration
│   ├── moe_loss.py            # Load balancing loss for MoE
│   └── indexer_loss.py        # Indexer alignment loss
└── finetune/                  # Fine-tuning utilities
    ├── __init__.py
    ├── grpo.py                # GRPO trainer wrapper
    └── grpo_algo.py           # GRPO algorithm implementation
```

## File Descriptions

### Root Level Files

#### `model.py`
The main model architecture file containing:
- **`ModelConfig`**: Dataclass holding all model hyperparameters
  - Architecture params (layers, dimensions, heads)
  - MLA configuration (latent dimensions, RoPE settings)
  - MoE configuration (experts, routing strategy)
- **`DeepSeekV3`**: Main model class
  - Embedding layer
  - Stack of decoder blocks
  - Output projection layer
  - Forward pass with KV caching support

**Key Features**:
- Supports both training and inference modes
- KV cache management for efficient generation
- Returns auxiliary outputs for loss computation (attention weights, gate logits, etc.)

#### `inference.py`
Inference utilities for text generation:
- **`generate()`**: Standard LLM generation function
  - Prefill phase: Process entire prompt with caching
  - Decode phase: Autoregressive token generation
  - Supports temperature, top-k, and top-p sampling
- **`sample_token()`**: Token sampling with various strategies

**Usage Pattern**:
```python
output = generate(
    model, tokenizer, 
    prompt="Hello, world!",
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9
)
```

#### `test_model.py`
Comprehensive model testing script that validates:
- Training mode (no cache, with auxiliary outputs)
- Prefilling mode (cache initialization)
- Decoding mode (incremental generation with cache)

### `deepseek_utils/` - Core Components

#### `attention.py`
Contains the attention mechanism implementations:

1. **`ScaledDotProductAttention`**
   - Standard scaled dot-product attention
   - Supports causal masking

2. **`Indexer`** (DeepSeek Sparse Attention)
   - Computes relevance scores for KV cache entries
   - Selects top-k most relevant tokens
   - Uses separate lightweight attention mechanism
   - Parameters: `num_heads`, `head_dim`, `rope_dim`, `k` (top-k)

3. **`MLA`** (Multi-head Latent Attention)
   - Compresses KV cache using low-rank projections
   - Two modes:
     - **MHA-MLA**: Multi-head for prefilling/training
     - **MQA-MLA**: Multi-query for decoding/inference
   - Integrates RoPE (Rotary Position Embedding)
   - Manages three cache tensors: `kv`, `k_rope`, `k_idx`

**Architecture Flow**:
```
Input → Indexer (select top-k) → MLA (attend to selected) → Output
```

#### `decoder.py`
Decoder block implementation:
- **`Decoder`**: Single transformer decoder layer
  - Pre-layer normalization
  - MLA attention sublayer
  - MoE/MLP feedforward sublayer
  - Residual connections
  - Conditional MoE usage (only after first N dense layers)

**Structure**:
```
x → LayerNorm → MLA → Residual
  → LayerNorm → MoE/MLP → Residual → output
```

#### `moe.py`
Mixture of Experts implementation:

1. **`SwiGLU`**
   - Swish-Gated Linear Unit activation
   - Formula: `SwiGLU(x) = SiLU(xW_gate) * (xW_value)`

2. **`CentroidRouter`**
   - Centroid-based expert routing
   - Computes cosine similarity to learned centroids
   - Temperature-scaled logits

3. **`MLP`**
   - Standard feedforward network
   - Uses SwiGLU activation
   - Used in dense (non-MoE) layers

4. **`MoE`**
   - Mixture of Experts layer
   - Shared experts (always active)
   - Routed experts (top-k selection)
   - Two router types: `linear` or `centroid`

**MoE Flow**:
```
Input → Router (select top-k experts)
      → Shared Experts (always active)
      → Weighted combination → Output
```

#### `utils.py`
Utility functions:
- **`pre_compute_theta()`**: Precompute RoPE frequency bases
- **`rotary_emb()`**: Apply rotary position embeddings

### `train/` - Training Infrastructure

#### `trainer.py`
Complete training framework:

1. **`Config`**: Training configuration dataclass
   - Device and precision settings
   - Optimizer configuration (AdamW)
   - Learning rate scheduler (CosineAnnealingLR)
   - Data loading parameters
   - Checkpoint management

2. **`Trainer`**: Training orchestrator
   - **`prep_dataset()`**: Initialize data loaders
   - **`train_epoch()`**: Single epoch training loop
   - **`validate()`**: Validation loop
   - **`train()`**: Full training procedure
   - **`save_checkpoints()`**: Save model state
   - **`load_checkpoints()`**: Load model state

**Training Loop**:
- Mixed precision training (AMP)
- Gradient scaling
- Auxiliary loss computation (MoE load balancing + indexer alignment)
- Progress tracking with tqdm

#### `moe_loss.py`
Load balancing auxiliary loss:
- **`compute_load_balancing_loss()`**
  - Encourages balanced expert usage
  - Computes token fraction per expert (f_i)
  - Computes average gate probability (P_i)
  - Loss: encourages uniform distribution

**Purpose**: Prevents expert collapse (all tokens routed to few experts)

#### `indexer_loss.py`
Indexer alignment loss:
- **`compute_indexer_loss()`**
  - Two modes:
    - **Warmup**: Align indexer distribution with full attention
    - **Training**: Align only on selected top-k tokens
  - Uses KL divergence between distributions

**Purpose**: Trains the indexer to select relevant tokens

### `finetune/` - Fine-tuning Tools

#### `grpo.py`
GRPO (Group Relative Policy Optimization) trainer wrapper:
- **`train()`**: Setup GRPO fine-tuning
  - Dataset loading
  - Reward function configuration
  - Training configuration

**Use Case**: Reinforcement learning from human feedback (RLHF)

#### `grpo_algo.py`
GRPO algorithm implementation (placeholder for custom GRPO logic)

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**:
- `torch==2.2.0` - PyTorch deep learning framework
- `tqdm` - Progress bars
- `trl` - Transformer Reinforcement Learning
- `datasets` - HuggingFace datasets
- `transformers` - HuggingFace transformers

## Quick Start

### Testing the Model

```bash
python test_model.py
```

This will run tests for:
- Training mode (with auxiliary outputs)
- Prefilling with cache
- Autoregressive decoding

### Training

```python
from model import DeepSeekV3, ModelConfig
from train.trainer import Trainer, Config

# Initialize model
model_config = ModelConfig()
model = DeepSeekV3(model_config)

# Setup training
train_config = Config()
trainer = Trainer(model, train_ds, val_ds, train_config)

# Train
trainer.train()
```

### Inference

```python
from model import DeepSeekV3, ModelConfig
from inference import generate

model = DeepSeekV3(ModelConfig()).cuda()
tokenizer = ...  # Your tokenizer

output = generate(
    model, tokenizer,
    prompt="Explain quantum computing:",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)
print(output)
```

## Key Concepts

### Multi-head Latent Attention (MLA)
- Reduces KV cache size through low-rank compression
- Separate latent spaces for queries, keys, and values
- Combines RoPE and non-RoPE components

### DeepSeek Sparse Attention (DSA)
- Indexer selects top-k relevant tokens from KV cache
- Reduces computational cost for long sequences
- Maintains quality while improving efficiency

### Mixture of Experts (MoE)
- Conditional computation: only activate subset of parameters
- Shared experts + routed experts
- Load balancing to prevent expert collapse

### KV Cache Management
Three cache tensors:
- `kv`: Compressed key-value latent representations
- `k_rope`: RoPE-applied key components
- `k_idx`: Indexer's selected token indices

## Model Configuration

Default configuration (see `ModelConfig` in `model.py`):
- Vocabulary: 1000 tokens
- Layers: 27 (3 dense + 24 MoE)
- Hidden dim: 4096
- Attention heads: 128
- MoE experts: 256 (16 shared + 240 routed)
- Top-k experts: 64
- Indexer top-k: 256 tokens

## Testing

Each component has dedicated test files:
- `deepseek_utils/test_attention.py` - Attention mechanisms
- `deepseek_utils/test_decoder.py` - Decoder blocks
- `deepseek_utils/test_moe.py` - MoE layers

## References

```bibtex
@misc{deepseekai2024deepseekv32,
  title={DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention}, 
  author={DeepSeek-AI},
  year={2025},
}
```

## Disclaimer

This is an **educational implementation** created for learning purposes. It may not include all optimizations and features of the official DeepSeek V3.2 model. For production use, please refer to the official DeepSeek repository.

## 📝 License

This educational project is provided as-is for learning purposes.

---
