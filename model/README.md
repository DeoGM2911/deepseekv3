# DeepSeekV3 architecture reconstruction

The DeepSeekV3 architecture is a decoder-only transformer-based model. The overall structure is similar to other LLMs. However,
the model implements some more complex components such as multi-head latent attention and mixture of experts.

## I. Components

### 1. Multi-head latent attention

The multi-head latent attention is a variant of multi-head self-attention. The underlying idea is that at inference time, in normal MHA, we need to
cache the keys and values of all previous tokens. This means that it's both memory-inefficient and computationally heavy. To address this, 
the DeepSeek team proprosed a new way to store the key-value cache: multi-head latent attention.

Now, instead of remembering the keys and values of all previous tokens, we only need to remember the latent "compressed" representation of keys and values and this latent representation is shared across all heads. The latent-dimension idea is one of the core ideas in a lot of ML models, and the aim here is to let the model learn how to compress the keys and values of all previous tokens into a latent space. This will save a lot of space, improve the inference speed, and in fact, improve the model performance.

As an example, in normal MHA, say we have 128 heads and 128 key-value dimension per head. Then we need to store 2 * 128 * 128 * L numbers in memory.
On the other hand, in latent MHA, we only need to store 576 * L for DeepSeekV3 numbers in memory, which is considerably lower than MHA. Of course,
we have other variants of MHA such as multi-head attention with one shared key-value cache or group multi-head attention. Both yield a lower key-value storage than MHA, but it limits the model expressiveness.

As an illsutration, below is a diagram describing MLA given the input $\bold{x}$ and the previous latent vector $\bold{z}$ (which can be `None` when training):

$$
\begin{aligned}
    &\text{Compute this input's latent representation: } &\bold{x}W_{xz} = \bold{z}_x & \\
    &\text{Update the current KV-cache: } &\bold{z} = \operatorname{concat}(\bold{z}, \bold{z}_x) & \\
    &\text{Compute keys \& values: } &\bold{z}W_{zv} = \bold{v}; \bold{z}W_{zk} = \bold{k} & \\
    &\text{Perform attention: } &\bold{o} = \operatorname{softmax}\bigg(\dfrac{\bold{x}W_q\bold{k}^T}{\sqrt{d_k}}\bigg)\bold{v}W_o & \\
\end{aligned}
$$

There's also the implementation of RoPE position encoding, but this will be mentioned later.

Other than that, the model uses pre-LayerNorm with residual connections.

### 2. Mixture of experts

Besides MLA, DeepSeekV3 also implements Mixture of Experts (MoE). MoE is a technique that allows the model to learn how to distribute the load of computing the attention between different experts. Each expert is a simple feed-forward network.

Here, the router of the MoE layer is implemented with a centroid router. A centroid router is a router that send the input tensor to the centroid of the closest expert. It's like 1-nearest neighbor but for choosing which expert to send the input tensor to. The implementation also allows for linear gating, but the default choice is centroid.

In addition, the MoE also has one shared expert whose outputs will always be included. In the end, the shared expert's outputs and the top-k experts' outputs will be combined to form the final output.

One note is that the activation function used in the MoE layer is SwiLU, which is the following:

$$\operatorname{SwiLU}(x) = \operatorname{SiLU}(W_gx) \odot W_vx$$
where $W_g$ and $W_v$ are learnable parameters. $W_g$ is the gate projection, and $W_v$ is the value projection. 

One small exception is that DeepSeekV3 first 3 decoder blocks are implemented with normal Position FFN instead of MoE.

### 3. Positional Encoding

For DeepSeekV3, the positional encoding is implemented with RoPE (Rotary Position Embedding). In short, RoPE helps to encode the relative position of the tokens in the sequence. This is more semantically rich compared to absolute positional encoding since in most cases, the relative position is more important than the absolute position.

One way to understand this is to imagine the token embeddings is rotated in there high-dimensional spaces by RoPE, and the angle between the token vectors will be the same as long as their relative position is the same.

## II. Implementation

Here, I implement the model using PyTorch. The breakdown of the files in this folder is as follow:

- [attention.py](./attention.py): This module implements the scaled dot product attention mechanism. Then extends it to the MLA one. Note that the KV-cache is returned if needed.
- [MoE.py](./MoE.py): This module implements the Mixture of Experts (MoE) layer. It uses a centroid router and allows for linear gating. It 
also implements the SwiLU activation function.
- [residual.py](./residual.py): This module implements the residual connection with pre-LayerNorm.
- [transformer_decoder.py](./transformer_decoder.py): This module implements the transformer decoder. It uses the residual connection, the attention layer, and the MoE/FFN layer.
- [utils.py](./utils.py): This module implements the absolute positional encoding and the RoPE positional encoding.
- [deepseek.py](./deepseek.py): This module implements the DeepSeekV3 model which is built from smaller decoder blocks. It also implements a generate function with 2 modes: `greedy` and `beam_search`. Note that `beam_search` allows for sampling for RLHF/DPO.
