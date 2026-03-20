import os
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader

from config import GPTConfig
from model import GPTStyleTransformer
from data import GPTStreamingDataset

def main():
    config = GPTConfig()
    
    # 1. Setup device and precision
    print(f"Using device: {config.device}")
    
    # 2. Setup Dataset
    print(f"Loading {config.dataset_name} ({config.dataset_subset})...")
    hf_ds = load_dataset(
        config.dataset_name,
        name=config.dataset_subset,
        split="train",
        streaming=True
    )
    encoder = tiktoken.get_encoding("cl100k_base")
    
    hf_iterator = iter(hf_ds)
    
    def create_dataloader(block_size, batch_size):
        stride = block_size // 2
        train_dataset = GPTStreamingDataset(
            hf_iterator=hf_iterator,
            encoder=encoder,
            block_size=block_size,
            stride=stride
        )
        return DataLoader(train_dataset, batch_size=batch_size)
        
    current_block_size = config.curriculum_schedule[0]["block_size"]
    current_batch_size = config.curriculum_schedule[0]["batch_size"]
    train_loader = create_dataloader(current_block_size, current_batch_size)
    data_iter = iter(train_loader)
    
    # 3. Initialize Model
    model = GPTStyleTransformer(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {total_params:,} trainable parameters.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 4. Training Loop setup
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    model.train()
    
    print(f"\nStarting Training! Committing to 5 Billion Tokens ({config.max_steps} steps).")
    step = 0
    while True:
        # Check for curriculum upgrades
        if step in config.curriculum_schedule and step != 0:
            new_params = config.curriculum_schedule[step]
            current_block_size = new_params["block_size"]
            current_batch_size = new_params["batch_size"]
            print(f"\n*** Curriculum Upgrade at Step {step} ***")
            print(f"New Block Size: {current_block_size}, New Batch Size: {current_batch_size}")
            train_loader = create_dataloader(current_block_size, current_batch_size)
            data_iter = iter(train_loader)
            
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            print("Dataset stream exhausted.")
            break
            
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)
        
        # Forward pass
        logits = model(x_batch)
        
        # Loss calculation requires (Batch * Time, Vocab Size) shape
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size), 
            y_batch.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True) # more efficient than simple zero_grad()
        loss.backward()
        
        # Gradient clipping stabilizes training for Transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step:07d} | Loss: {loss.item():.4f}")
            
        step += 1
            
        # Unified Stopping mechanism for precisely 5 Billion Tokens
        if step >= config.max_steps:
            print(f"\nTarget Reached: Processed 5 Billion tokens (Step {step}). Stopping training!")
            checkpoint_path = os.path.join(config.checkpoint_dir, "gpt_model_5B_tokens.pt")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'loss': loss.item(),
            }, checkpoint_path)
            
            print(f"Model successfully saved to {checkpoint_path}")
            break

if __name__ == "__main__":
    main()
