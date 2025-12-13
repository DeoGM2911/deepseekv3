import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import PosEncoding
from .transformer_decoder import DecoderBlock


class AnswerHead(nn.Module):
    """
    A simple linear layer for answer head (LM head).
    """
    def __init__(
        self,
        input_dim,
        vocab_size
    ):
        super(AnswerHead, self).__init__()
        self.lm_head = nn.Linear(input_dim, vocab_size)
    
    def forward(self, x):
        return self.lm_head(x)


class DeepSeekV3(nn.Module):
    def __init__(
        self,
        vocab_size,
        input_dim,
        latent_dim=576,
        max_len=1000,
        num_layers=8,
        num_heads=8,
        moe=True,
        k=8,
        ffn_hidden_dim=1024,
        num_experts=16,
        router="centroid",
        pos_enc="rotary",
        dropout=0.1
    ):
        super(DeepSeekV3, self).__init__()

        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.pos_enc_type = pos_enc

        # Positional encoding (if not using rotary)
        if pos_enc == "absolute":
            self.pos_enc = PosEncoding(max_len, input_dim)
        elif pos_enc == "learnable":
            self.pos_enc = nn.Parameter(torch.zeros(1, max_len, input_dim))
        else:
            self.pos_enc = None
        
        # First 3 decoder blocks (or fewer if num_layers < 3) use normal FFN
        num_dense = min(3, num_layers)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                input_dim,
                latent_dim,
                num_heads,
                num_experts,
                ffn_hidden_dim,
                moe=False,
                k=k,
                rotary_enc=(pos_enc == "rotary"),
                router=router,
                dropout=dropout
            )
            for _ in range(num_dense)
        ])

        # The remaining decoder blocks use MoE
        self.decoder_blocks.extend(nn.ModuleList([
            DecoderBlock(
                input_dim,
                latent_dim,
                num_heads,
                num_experts,
                ffn_hidden_dim,
                moe=True,  # Use parameter
                k=k,      # Use parameter
                rotary_enc=(pos_enc == "rotary"),
                router=router,  # Use parameter
                dropout=dropout  # Use parameter
            )
            for _ in range(num_layers - num_dense)
        ]))

        # The answer head (LM head) - uses input_dim, not latent_dim
        self.answer_head = AnswerHead(input_dim, vocab_size)
        
    def forward(
        self,
        x,
        cache=False,
        kv_cache=None,
        valid_lens=None
    ):
        # KV cache
        if cache:
            kv_cache_memory = []
        else:
            kv_cache_memory = None
        
        if kv_cache is None:
            kv_cache = [None] * self.num_layers
        else:
            # Unpack tensor to list for iteration
            kv_cache = list(kv_cache.unbind(dim=0))
        
        # MoE logits for auxilary loss
        moe_logits_list = []
        moe_topk_indices_list = []

        # Embedding
        x = self.embedding(x)

        # Apply positional encoding if needed
        if self.pos_enc is not None:
            if self.pos_enc_type == "learnable":
                # For learnable embeddings, add them
                seq_len = x.size(1)
                x = x + self.pos_enc[:, :seq_len, :]
            else:
                # For absolute positional encoding
                x = self.pos_enc(x)

        # Decoder blocks - capture all return values
        for i, block in enumerate(self.decoder_blocks):
            x, _, _, moe_logits, moe_topk_indices, new_kv_cache = block(x, kv_cache[i], valid_lens)
            
            if cache:
                kv_cache_memory.append(new_kv_cache)
            
            moe_logits_list.append(moe_logits) if moe_logits is not None else None
            moe_topk_indices_list.append(moe_topk_indices) if moe_topk_indices is not None else None
        
        if cache and kv_cache_memory is not None:
            kv_cache_tensor = torch.stack(kv_cache_memory, dim=0)
        else:
            kv_cache_tensor = None

        # Stack the moe logits and indices
        moe_logits_list = torch.stack(moe_logits_list, dim=0) if moe_logits_list else None
        moe_topk_indices_list = torch.stack(moe_topk_indices_list, dim=0) if moe_topk_indices_list else None

        # Answer head
        return self.answer_head(x), (moe_logits_list, moe_topk_indices_list), kv_cache_tensor
    
    def _top_k_search(
        self,
        prompt_tokens,
        max_new_tokens=100,
        eos_token_id=None,
        pad_token_id=0,
        temperature=1.0,
        top_k=None,
        include_prompt=True,
        valid_lens=None
    ):
        """
        Autoregressive generation with KV-caching
        
        Args:
            prompt_tokens: (batch, seq_len) - input prompt token IDs
            max_new_tokens: maximum number of tokens to generate
            eos_token_id: end-of-sequence token ID
            temperature: sampling temperature
            top_k: if set, only sample from top k tokens
        
        Returns:
            generated_tokens: (batch, seq_len + max_new_tokens)
        
        """
        # Track which sample is complete
        complete = torch.zeros(prompt_tokens.size(0), dtype=torch.bool, device=prompt_tokens.device)

        # kv_cache
        kv_cache_memory = None
        
        # Step 1: Process the ENTIRE prompt once
        logits, _, kv_cache_memory = self.forward(prompt_tokens, cache=True, valid_lens=valid_lens)  # (batch, prompt_len, vocab_size)
        
        # Get last token's logits for first generation
        next_token_logits = logits[:, -1, :] / temperature  # (batch, vocab_size)
        
        # Sample first new token
        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits.scatter_(1, top_k_indices, top_k_logits)
        
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        
        # Check if first token is EOS
        if eos_token_id is not None:
            complete = (next_token.squeeze(-1) == eos_token_id)
        
        # Start building generated sequence
        if include_prompt:
            generated = torch.cat([prompt_tokens, next_token], dim=1)
        else:
            generated = next_token
        
        # Step 2+: Generate remaining tokens ONE AT A TIME
        for _ in range(max_new_tokens - 1):
            # For completed sequences, use pad token instead of running forward pass
            # Create a copy of next_token for forward pass
            next_token_for_forward = next_token.clone()
            next_token_for_forward[complete] = pad_token_id
            
            # Process ONLY the last generated token (KV cache handles the rest!)
            logits, _, kv_cache_memory = self.forward(next_token_for_forward, cache=True, kv_cache=kv_cache_memory, valid_lens=None)  # (batch, 1, vocab_size)
            next_token_logits = logits[:, -1, :] / temperature  # (batch, vocab_size)
            
            # Optional top-k sampling
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample next token (only for incomplete sequences)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # For completed sequences, use pad token
            next_token[complete] = pad_token_id
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS token in newly generated tokens
            if eos_token_id is not None:
                # Mark sequences as complete if they generated EOS (and weren't already complete)
                newly_complete = (next_token.squeeze(-1) == eos_token_id) & (~complete)
                complete = complete | newly_complete
            
            # Stop if all sequences are complete
            if complete.all() or generated.size(1) >= max_new_tokens:
                break
        
        return generated
    
    def _beam_search(
        self,
        prompt_tokens,
        max_new_tokens=100,
        eos_token_id=None,
        pad_token_id=0,
        temperature=1.0,
        beam_size=8,
        length_penalty=1.0,
        include_prompt=True,
        valid_lens=None,
        do_sample=False
    ):
        """
        Beam search generation with KV-caching.
        
        Args:
            prompt_tokens: (batch_size, prompt_len) - input prompts
            max_new_tokens: maximum tokens to generate
            eos_token_id: end-of-sequence token ID
            pad_token_id: padding token ID
            temperature: sampling temperature
            beam_size: number of beams to maintain
            length_penalty: exponent for length normalization (0.6-1.0)
            include_prompt: whether to include prompt in output
            
        Returns:
            best_sequences: (batch_size, seq_len) - best sequence per batch
        """
        
        # ============================================================
        # STEP 1: INITIALIZATION
        # ============================================================
        # Get batch_size and device from prompt_tokens
        batch_size = prompt_tokens.size(0)
        device = prompt_tokens.device
        
        # KV cache
        kv_cache_memory = None
        
        # ============================================================
        # STEP 2: PROCESS PROMPT & GET INITIAL BEAMS
        # ============================================================
        # Forward pass through the model with the prompt
        logits, _, kv_cache_memory = self.forward(prompt_tokens, cache=True, kv_cache=kv_cache_memory, valid_lens=valid_lens)  # Shape: (batch_size, prompt_len, vocab_size)
        
        # Get logits for the last token and apply temperature
        vocab_size = logits.size(-1)
        next_token_logits = logits[:, -1, :] / temperature  # Shape: (batch_size, vocab_size)
        
        # Convert logits to probabilities
        probs = F.softmax(next_token_logits, dim=-1)  # Shape: (batch_size, vocab_size)
        
        # Get top-k (beam_size) initial tokens and their probabilities
        if not do_sample:
            init_scores, init_indices = torch.topk(probs, k=beam_size, dim=-1)  # Both: (batch_size, beam_size)
        else:
            init_indices = torch.multinomial(probs, num_samples=beam_size)
            init_scores = torch.gather(probs, dim=-1, index=init_indices)    
        
        # Convert probabilities to log probabilities for scores
        init_log_scores = torch.log(init_scores)  # Shape: (batch_size, beam_size)
        
        # ============================================================
        # STEP 3: CREATE INITIAL BEAM STATE
        # ============================================================
        # Pre-allocate sequence buffer for efficiency (avoid repeated concatenations)
        prompt_len = prompt_tokens.size(1)
        max_seq_len = prompt_len + max_new_tokens
        
        # Pre-allocate buffer: (batch_size, beam_size, max_seq_len)
        sequences = torch.zeros(
            (batch_size, beam_size, max_seq_len),
            dtype=torch.long,
            device=device
        )
        sequences = sequences.fill_(pad_token_id)
        
        # Fill in initial sequences
        if include_prompt:
            # Copy prompt to all beams
            sequences[:, :, :prompt_len] = prompt_tokens.unsqueeze(1).expand(batch_size, beam_size, -1)
            # Add first generated token
            sequences[:, :, prompt_len] = init_indices
            current_len = prompt_len + 1
        else:
            # Just the first generated token
            sequences[:, :, 0] = init_indices
            current_len = 1
        
        # Expand KV caches for all beams
        kv_cache_memory = kv_cache_memory.unsqueeze(2).expand(-1, -1, beam_size, -1, -1)
        # Now: (num_layers, batch_size, beam_size, seq_len, latent_dim)
        kv_cache_memory = kv_cache_memory.reshape(kv_cache_memory.size(0), batch_size * beam_size, -1, kv_cache_memory.size(-1))

        # Initialize beam search state
        scores = init_log_scores
        complete = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device)
        
        
        # ============================================================
        # STEP 4: PREPARE FOR GENERATION LOOP
        # ============================================================
        # Get the last token from each beam for next forward pass
        next_tokens = init_indices.view(batch_size * beam_size, 1)  # Shape: (batch_size * beam_size, 1)
        
        
        
        # ============================================================
        # STEP 5: GENERATION LOOP
        # ============================================================
        # Loop for max_new_tokens - 1 (we already generated first token)
        for _ in range(max_new_tokens - 1):
        
            # --------------------------------------------------------
            # 5.1: Check early stopping
            # --------------------------------------------------------
            # Break if all beams are complete
            if complete.all():
                break
            
            
            # --------------------------------------------------------
            # 5.2: Forward pass for next token (skip completed beams)
            # --------------------------------------------------------
            complete_flat = complete.view(-1)  # Flatten complete to (batch_size * beam_size,)
            incomplete_mask = ~complete_flat
            
            # Only process incomplete beams
            if incomplete_mask.any():
                # Extract tokens for incomplete beams only
                active_tokens = next_tokens[incomplete_mask]
                
                # Extract KV-cache for incomplete beams
                active_kv_cache = kv_cache_memory[:, incomplete_mask, :, :]
                
                # Forward pass only for active beams
                active_logits, _, active_kv_cache_new = self.forward(
                    active_tokens,
                    cache=True,
                    kv_cache=active_kv_cache, 
                    valid_lens=None
                )  # Shape: (num_active_beams, 1, vocab_size)
                
                # Reconstruct full KV-cache (can't do in-place due to seq_len growth)
                # Create new tensor with updated sequence length
                num_layers = kv_cache_memory.size(0)
                new_seq_len = active_kv_cache_new.size(2)
                latent_dim = kv_cache_memory.size(3)
                
                new_kv_cache_memory = torch.zeros(
                    (num_layers, batch_size * beam_size, new_seq_len, latent_dim),
                    dtype=kv_cache_memory.dtype,
                    device=device
                )
                
                # Copy updated caches for active beams
                new_kv_cache_memory[:, incomplete_mask, :, :] = active_kv_cache_new
                
                # For completed beams, copy old cache (pad with zeros for new seq_len)
                if complete_flat.any():
                    old_seq_len = kv_cache_memory.size(2)
                    new_kv_cache_memory[:, complete_flat, :old_seq_len, :] = kv_cache_memory[:, complete_flat, :, :]
                
                kv_cache_memory = new_kv_cache_memory
                
                # Create full logits tensor (fill completed beams with zeros, they won't be used)
                logits = torch.zeros(
                    (batch_size * beam_size, 1, active_logits.size(-1)),
                    dtype=active_logits.dtype,
                    device=device
                )
                logits[incomplete_mask] = active_logits
            else:
                # All beams complete, break early
                break
            
            # Get logits for last position and apply temperature
            next_token_logits = logits[:, -1, :] / temperature  # Shape: (batch_size * beam_size, vocab_size)
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=1)  # Shape: (batch_size * beam_size, vocab_size)
            
            
            
            # --------------------------------------------------------
            # 5.3: Compute candidate scores
            # --------------------------------------------------------
            # Reshape probabilities to (batch_size, beam_size, vocab_size)
            probs_reshaped = probs.view(batch_size, beam_size, -1)
            
            # Compute log probabilities
            log_probs = torch.log(probs_reshaped)
            
            # Add current beam scores (broadcast across vocab dimension)
            # scores is (batch_size, beam_size)
            # Need to make it (batch_size, beam_size, 1) for broadcasting
            scores = scores.view(batch_size, beam_size, 1)
            candidate_scores = scores + log_probs
            
            # For completed beams, set all candidate scores to -inf
            # except for the pad_token_id which should keep the current score
            # This prevents completed beams from changing
            if complete.any():
                candidate_scores[complete.view(batch_size, beam_size, 1).expand(-1, -1, vocab_size)] = -float('inf')
                candidate_scores[complete, pad_token_id] = scores[complete].squeeze(-1)
            
            
            
            # --------------------------------------------------------
            # 5.4: Select top-k beams
            # --------------------------------------------------------
            # Flatten candidate_scores to (batch_size, beam_size * vocab_size)
            candidate_scores_flat = candidate_scores.view(batch_size, -1)
            
            # Get top beam_size candidates for each batch
            top_scores, top_indices = torch.topk(candidate_scores_flat, beam_size, dim=1)
            
            # Determine which beam each candidate came from
            parent_beam_idx = top_indices // vocab_size  # (batch_size, beam_size)
            
            # Determine which token was selected
            selected_tokens = top_indices % vocab_size  # (batch_size, beam_size)
            
            # --------------------------------------------------------
            # 5.5: Update beam sequences (in-place)
            # --------------------------------------------------------
            # Gather parent sequences using parent_beam_idx
            # Use torch.gather or advanced indexing
            parent_sequences = torch.gather(sequences, 1, parent_beam_idx.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))
            
            # Update sequences buffer in-place
            sequences = parent_sequences
            sequences[:, :, current_len] = selected_tokens
            current_len += 1

            # Update KV cache - (num_layers, batch_size * beam_size, seq_len, latent_dim)
            # Reshape, gather, reshape back to (num_layers, batch_size * beam_size, seq_len, latent_dim)
            num_layers = kv_cache_memory.size(0)
            seq_len = kv_cache_memory.size(2)
            latent_dim = kv_cache_memory.size(3)

            reshaped_kv = kv_cache_memory.view(num_layers, batch_size, beam_size, seq_len, latent_dim)
            # Expand parent_beam_idx to match dimensions
            idx = parent_beam_idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, batch, beam, 1, 1)
            idx = idx.expand(num_layers, -1, -1, seq_len, latent_dim)
            # Gather along beam dimension (dim=2)
            new_kv_cache_memory = torch.gather(reshaped_kv, 2, idx)
            # Flatten back to (num_layers, batch*beam, seq_len, latent)
            kv_cache_memory = new_kv_cache_memory.view(num_layers, batch_size * beam_size, seq_len, latent_dim)


            # --------------------------------------------------------
            # 5.6: Update completion status
            # --------------------------------------------------------
            # Gather parent completion status by checking which sequences end with eos token
            parent_complete = torch.gather(complete, 1, parent_beam_idx)
            
            # Check if newly selected tokens are EOS
            if eos_token_id is not None:
                newly_complete = (selected_tokens == eos_token_id)
                new_complete = parent_complete | newly_complete
            else:
                new_complete = parent_complete
            
            
            # --------------------------------------------------------
            # 5.7: Update beam state
            # --------------------------------------------------------
            # Update state with new values (sequences already updated in-place)
            scores = top_scores
            complete = new_complete
            
            
            # Prepare next_tokens for next iteration
            # Flatten sequences for next forward pass
            next_tokens = selected_tokens.view(batch_size * beam_size, -1)
        
        
        # ============================================================
        # STEP 6: SELECT BEST SEQUENCES
        # ============================================================
        # Trim sequences to actual length (remove unused buffer space)
        sequences = sequences[:, :, :current_len]
        
        # Apply length normalization if needed
        if length_penalty != 1.0:
            # Manual length normalization
            lengths = (sequences != 0).sum(dim=-1)  # (batch_size, beam_width)
            final_scores = scores / (lengths.float() ** length_penalty)
        else:
            final_scores = scores
        
        # Get best sequence for each batch
        best_scores, best_indices = torch.max(final_scores, dim=1) # (batch, seq_len)
        best_sequences = sequences[torch.arange(batch_size), best_indices]


        # Return best sequences
        return best_sequences, best_scores
        
    
    def generate(
        self, 
        prompt_tokens,
        decode_strat="beam", 
        max_new_tokens=100, 
        eos_token_id=None, 
        pad_token_id=0, 
        temperature=1.0, 
        top_k_or_beam_size=None, 
        include_prompt=True,
        valid_lens=None,
        do_sample=False
    ):
        if decode_strat == "beam":
            return self._beam_search(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                temperature=temperature,
                beam_size=top_k_or_beam_size,
                include_prompt=include_prompt,
                valid_lens=valid_lens
            )
        elif decode_strat == "top_k":
            return self._top_k_search(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                temperature=temperature,
                top_k=top_k_or_beam_size,
                include_prompt=include_prompt,
                valid_lens=valid_lens,
                do_sample=do_sample
            )
        else:
            raise ValueError(f"Invalid decode strategy: {decode_strat}")


if __name__ == "__main__":
    dummy_vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9}
    dummy_input = torch.randint(0, 10, (2, 10))  # batch_size=2
    input_dim = 256
    
    # Create model
    deepseek = DeepSeekV3(
        vocab_size=len(dummy_vocab),
        input_dim=input_dim,
        num_layers=4,
        pos_enc="rotary"
    )
    
    # Test forward pass
    output = deepseek(dummy_input)
    print(f"Output shape: {output[0].shape}")  # Should be (2, 10, vocab_size)
    
    # Test generation
    generated = deepseek.generate(dummy_input, max_new_tokens=5, top_k_or_beam_size=3)
    print(f"Generated: {generated}")
    print(f"Generated shape: {generated.shape}")  # Should be (2, 15)
