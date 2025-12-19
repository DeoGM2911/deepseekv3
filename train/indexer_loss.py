import torch
import torch.nn.functional as F


def compute_indexer_loss(
    unmasked_scores, 
    indexer_scores, 
    indexer_indices, 
    mask, 
    warmup=True
):
    """
    Compute the indexer loss. There are two modes: warm-up stage where we align the indexer distribution
    with the attention distribution and the training stage where we align with only with respect to the
    indexer's choosen tokens.

    @Args:
        unmasked_scores: (batch, seq_len, vocab_size) - unmasked scores from the dense attention mechanism
        indexer_scores: (batch, seq_len, num_experts) - indexer scores from the indexer
        indexer_indices: (batch, seq_len) - indexer indices from the indexer
        mask: (batch, seq_len) - mask for the input. Only the causal and the padding mask - indexer mask not included.
        warmup: bool - whether to use warmup mode
    """
    # Stack along layers
    unmasked_scores = torch.stack(unmasked_scores, dim=0)
    indexer_scores = torch.stack(indexer_scores, dim=0)
    indexer_indices = torch.stack(indexer_indices, dim=0)

    # Detach the attention scores from the graph
    unmasked_scores = unmasked_scores.detach()

    # Apply causal and padding mask to unmasked scores
    attention_probs = unmasked_scores.masked_fill(mask == 0, -float('inf'))
    indexer_probs = F.softmax(indexer_scores, dim=-1)

    if not warmup:
        # Align with respect to the indexer's choosen tokens
        indexer_probs = indexer_probs.gather(-1, indexer_indices.unsqueeze(-1)).squeeze(-1)
        attention_probs = attention_probs.gather(-1, indexer_indices.unsqueeze(-1)).squeeze(-1)
    
    attention_probs = F.softmax(attention_probs, dim=-1)
    loss = F.kl_div(indexer_probs.log(), attention_probs, reduction='batchmean')
    
    return loss
    
