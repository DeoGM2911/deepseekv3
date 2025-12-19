import torch
from torch import optim, nn
from dataclasses import dataclass, field
from torch.amp import GradScaler, autocast
from .moe_loss import compute_load_balancing_loss
from .indexer_loss import compute_indexer_loss
from tqdm import tqdm
import os
from ..model import ModelConfig

@dataclass
class Config:
    # Training hyperparams - AdamW
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate: float = 1e-4
    num_epochs: int = 10
    mix_precision: bool = True
    weight_decay: float = 0.0001
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    precision: torch.dtype = torch.bfloat16

    # LR scheduler
    lr_scheduler: str = "CosineAnnealingLR"
    lr_scheduler_dict: dict = field(default_factory=lambda: {
        "StepLR": optim.lr_scheduler.StepLR,
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    })
    lr_scheduler_params: dict = field(default_factory=lambda: {
        "StepLR": {
            "step_size": 1,
            "gamma": 0.9,
        },
        "CosineAnnealingLR": {
            "T_max": None,  # Set by the trainer
        },
    })

    # Dataloader params
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    # Model hyperparams
    model_config: ModelConfig = ModelConfig()
    
    # Checkpoints
    save_dir: str = "checkpoints"
    save_interval: int = 1


class Trainer:
    def __init__(self, model, train_ds, val_ds, config: Config):
        self.model = model
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.prep_data = False
        self.epoch = 0
        self.train_loss = None
        self.val_loss = None

        assert self.config.num_layers > 3, "the number of layers must be at least 4"

        # Mixed pecision training
        if self.config.mix_precision:
            self.scaler = GradScaler(self.config.device)
        else:
            self.scaler = None

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        # Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Custom initialization for scheduler is in prep_dataset
        self.lr_scheduler = None

        # Create checkpoint dir
        if not os.path.exists(self.config.save_dir):
            print("Making sav dir: ", self.config.save_dir)
            os.makedirs(self.config.save_dir, exist_ok=True)
    
    def prep_dataset(self):
        """
        Prepare the dataloader for training
        """
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        # Initialize LR scheduler with correct steps
        total_steps = self.config.num_epochs * len(self.train_loader)
        if self.config.lr_scheduler == "CosineAnnealingLR":
            self.config.lr_scheduler_params["CosineAnnealingLR"]["T_max"] = total_steps

        self.lr_scheduler = self.config.lr_scheduler_dict[self.config.lr_scheduler](
            self.optimizer,
            **self.config.lr_scheduler_params[self.config.lr_scheduler],
        )

        self.prep_data = True

    def train_epoch(self):
        assert self.prep_data, "Please call prep_dataset() before training"
        self.model.train()
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            inputs, labels, masks = batch
            inputs, labels, masks = inputs.to(self.config.device), labels.to(self.config.device), masks.to(self.config.device)
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with autocast(self.config.device.type, dtype=self.config.precision):
                outputs, _, _, _, attn_weights, unmasked_scoers, indexer_indices, indexer_scores, more_logits_list, \
                more_topk_indices,  = self.model(inputs)
                moe_loss = compute_load_balancing_loss(moe_logits_list, moe_topk_indices, self.config.model_config.moe_num_experts)
                indexer_loss = compute_indexer_loss(attn_weights, indexer_indices, indexer_scores, masks)

                # Reshape for CrossEntropyLoss: (Batch * Seq, Vocab) vs (Batch * Seq)
                vocab_size = outputs.size(-1)
                loss = self.criterion(outputs.view(-1, vocab_size), labels.view(-1)) + moe_loss + indexer_loss
            
            if self.config.mix_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
            self.lr_scheduler.step()

    @torch.no_grad()
    def validate(self):
        assert self.prep_data, "Please call prep_dataset() before training"
        self.model.eval()
        pbar = tqdm(self.val_loader, desc="Validation")
        for batch in pbar:
            inputs, labels, masks = batch
            inputs, labels, masks = inputs.to(self.config.device), labels.to(self.config.device), masks.to(self.config.device)
            
            # Mixed precision training
            with autocast(self.config.device.type, dtype=self.config.precision):
                outputs, (moe_logits_list, moe_topk_indices), _ = self.model(inputs)
                moe_loss = compute_load_balancing_loss(moe_logits_list, moe_topk_indices, self.config.model_config.moe_num_experts)
                indexer_loss = compute_indexer_loss(outputs, masks)
                
                vocab_size = outputs.size(-1)
                loss = self.criterion(outputs.view(-1, vocab_size), labels.view(-1)) + moe_loss + indexer_loss
            
            pbar.set_postfix({"loss": loss.item()})
    
    def train(self):
        self.prep_dataset()
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self.train_epoch()
            self.validate()
            if epoch % self.config.save_interval == 0:
                self.save_checkpoints(os.path.join(self.config.save_dir, f"epoch_{epoch}.pt"))
    
    def save_checkpoints(self, checkpoint_path: str):
        """
        Save the model and optimizer state dict to a checkpoint file
        """
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }, checkpoint_path)
    
    def load_checkpoints(self, checkpoint_path: str, only_model: bool = False):
        """
        Load the model and optimizer state dict from a checkpoint file
        """
        if os.path.exists(checkpoint_path):
            if only_model:
                self.model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
            else:
                self.model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
                self.optimizer.load_state_dict(torch.load(checkpoint_path)["optimizer_state_dict"])
                self.epoch = torch.load(checkpoint_path)["epoch"]
                self.train_loss = torch.load(checkpoint_path)["train_loss"]
                self.val_loss = torch.load(checkpoint_path)["val_loss"]
        else:
            print(f"Checkpoint {checkpoint_path} not found!")
    