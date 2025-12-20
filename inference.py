import torch
from typing import Optional
from model import DeepSeekV3, ModelConfig


def generate(
    model: DeepSeekV3,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
    mask: Optional[torch.Tensor] = None
):
    """
    Standard LLM generate function for DeepSeekV3.

    Args:
        model: The DeepSeekV3 model instance.
        tokenizer: Tokenizer with encode() and decode() methods.
        prompt: Input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: If set, only sample from top-k tokens.
        top_p: If set, use nucleus sampling with this probability mass.
        eos_token_id: Stop generation when this token is generated.
        device: Device to run inference on.

    Returns:
        Generated text string (including the prompt).
    """
    model.eval()

    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids.clone()

    # Initialize KV cache as None (first forward pass will populate it)
    kv_cache = None
    k_rope_cache = None
    k_idx_cache = None

    with torch.no_grad():
        # Prefill phase: process the entire prompt
        logits, kv_cache, k_rope_cache, k_idx_cache, *_ = model(
            input_ids,
            cache=True,
            kv=None,
            k_rope=None,
            k_idx=None,
            mask=mask,
            inference=True
        )

        # Get logits for the last token
        next_token_logits = logits[:, -1, :]

        # Sample next token
        next_token = sample_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Check for EOS
        if eos_token_id is not None and next_token.item() == eos_token_id:
            return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Decode phase: generate tokens one at a time using cache
        for _ in range(max_new_tokens - 1):
            # Only feed the last generated token
            logits, kv_cache, k_rope_cache, k_idx_cache, *_ = model(
                next_token,
                cache=True,
                kv=kv_cache,
                k_rope=k_rope_cache,
                k_idx=k_idx_cache,
                mask=None,
                inference=True
            )

            next_token_logits = logits[:, -1, :]
            next_token = sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    # Decode and return the generated text
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Sample a token from logits with temperature, top-k, and top-p sampling.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size).
        temperature: Sampling temperature.
        top_k: If set, only sample from top-k tokens.
        top_p: If set, use nucleus sampling.

    Returns:
        Sampled token IDs of shape (batch_size, 1).
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Sample from the distribution
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
