import torch
import torch.nn as nn
from torch.utils.data import Dataset
from model.deepseek import DeepSeekV3
from train.trainer import Trainer, Config
import os

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, length=100, max_len=50):
        self.length = length
        self.max_len = max_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Random inputs and labels
        inputs = torch.randint(0, 1000, (self.max_len,))
        labels = torch.randint(0, 1000, (self.max_len,))
        return inputs, labels

def verify_trainer():
    print("Verifying Trainer implementation...")
    
    # Config
    config = Config(
        num_epochs=1,
        batch_size=4,
        embed_dim=64,
        latent_dim=64,
        num_heads=2,
        num_layers=6,
        max_len=50,
        ffn_hidden_dim=128,
        save_interval=1,
        device=torch.device("cpu"), # Use CPU for verification to avoid CUDA issues
        precision=torch.float32,    # Use float32 for CPU
        mix_precision=False         # Disable mix precision for CPU
    )
    
    # Model
    vocab_size = 1000
    model = DeepSeekV3(
        vocab_size=vocab_size,
        input_dim=config.embed_dim,
        latent_dim=config.latent_dim,
        max_len=config.max_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_hidden_dim=config.ffn_hidden_dim,
        pos_enc="rotary"
    )
    
    # Datasets
    train_ds = DummyDataset()
    val_ds = DummyDataset()
    
    # Trainer
    try:
        trainer = Trainer(model, train_ds, val_ds, config=config)
        print("Trainer initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Trainer: {e}")
        return

    # Train
    try:
        print("Starting training loop (1 epoch)...")
        trainer.train()
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nVerification Passed!")


def test_generate():
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

    print("\nTest passed!")

if __name__ == "__main__":
    test_generate()
    verify_trainer()
