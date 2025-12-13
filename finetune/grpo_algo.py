import torch
from torch import nn, optim
from ..model.deepseek import DeepSeekV3
from torch.utils.data import DataLoader
from copy import deepcopy


class GRPOLoss(nn.Module):
    def __init__(self, beta: float, num_output):
        super().__init__()
        self.beta = beta
        self.num_output = num_output

    def forward(self, advantages, model_probs, old_model_probs, ref_probs):
        # KL Divergence
        ref_ratio = (ref_probs / model_probs).view(-1, self.num_output)  
        kl_loss = self.beta * torch.mean(torch.mean(ref_ratio - torch.log(ref_ratio) - 1, dim=1))

        # Policy gradient
        old_ratio = (model_probs / old_probs).view(-1, self.num_output)
        pg_loss = torch.mean(torch.mean(advantages * old_ratio, dim=1))

        return -pg_loss + kl_loss  # Goal is to maximize


class GRPO:
    def __init__(
        self, 
        reward_model_path: str, 
        ref_model_path: str,
        model_path: str,
        dataloader: DataLoader,
        vocab,
        device: torch.device,
        update_steps:int,
        num_epochs: int,
        lr: float,
        weight_decay: float,
        num_output: int,
        kl_beta: float
    ):
        # Reward model
        self.reward_model = torch.load(reward_model_path, map_location=device)
        self.reward_model.to(device)
        self.reward_model.requires_grad_(False)

        # Reference model
        self.ref_model = torch.load(ref_model_path, map_location=device)  # Assume the whole model is remembered
        self.ref_model.to(device)
        self.ref_model.requires_grad_(False)

        # Model to be finetuned
        self.model = torch.load(model_path, map_location=device)
        self.model.to(device)
        self.model.requires_grad_(True)

        # Old model in GRPO
        self.old_model = deepcopy(self.model)
        self.old_model.to(device)
        self.old_model.requires_grad_(False)
        
        # Data loader
        self.vocab = vocab
        self.dataloader = dataloader
        self.device = device

        # Optimizer hyperparams
        self.update_steps = update_steps
        self.num_epochs = num_epochs
        self.num_output = num_output
        self.lr = lr
        self.weight_decay = weight_decay
        self.kl_beta = kl_beta

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = GRPOLoss(self.kl_beta, self.num_output)

    def _get_probs(self, model, questions, outputs, valid_lens):
        # For each token, compute the cumulative probability for the next token
        # Initial forward pass
        logits, _, cache = model(questions, cache=True, valid_lens=valid_lens)

        # Get the cumulative probability for each token
        probs = torch.softmax(logits, dim=-1)
        scores = torch.gather(probs, outputs[:, 0].view(probs.shape[0], -1), dim=-1)

        for i in range(1, outputs.shape[1]):
            logits, _, cache = model(outputs[:, i-1].view(questions.shape[0], -1), cache=cache)
            probs = torch.softmax(logits, dim=-1)
            next_probs = torch.gather(probs, outputs[:, i].view(probs.shape[0], -1), dim=-1)
            # Times 1 if it's pad token
            next_probs[outputs[:, i] == self.vocab.pad_token_id] = 1
            scores *= next_probs

        return scores

    def train(self):
        step = 0
        for epoch in range(self.num_epochs):
            for i, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                step += 1

                questions = batch['questions']
                valid_lens = batch['valid_lens']

                # Duplicate each questions num_output time for batch processing
                questions = questions.repeat(self.num_output, 1)

                # Forward pass to get outputs
                outputs, probs = self.model.generate(questions,
                                decode_strat="beam", 
                                max_new_tokens=1_000, 
                                eos_token_id=vocab.eos_token_id, 
                                pad_token_id=vocab.pad_token_id, 
                                temperature=1.0, 
                                top_k_or_beam_size=8, 
                                include_prompt=False,
                                valid_lens=valid_lens,
                                do_sample=True
                )

                # Forward pass for old model
                with torch.no_grad():
                    old_probs = self._get_probs(self.old_model, questions, outputs, valid_lens)

                # Forward pass to get ref outputs
                with torch.no_grad():
                    ref_probs = self._get_probs(self.ref_model, questions, outputs, valid_lens)

                # Compute the rewards
                with torch.no_grad():
                    rewards = self.reward_model(questions, outputs)
                
                # Compute the advantages - Normalize for every num_output items
                rewards = rewards.view(self.dataloader.batch_size, self.num_output)
                advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / rewards.std(dim=1, keepdim=True)  # (batch_size, num_output)

                # Compute the loss
                loss = self.criterion(advantages, probs, old_probs, ref_probs)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update old model
                if step % self.update_steps == 0:
                    self.old_model.load_state_dict(self.model.state_dict())
                    self.old_model.requires_grad_(False)
                    step = 0
