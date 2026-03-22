import os
import time
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
import warnings

# Let PyTorch use TensorFloat32 on RTX 4090s globally!
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from dotenv import load_dotenv

# Explicitly load local .env variables
load_dotenv()

from config import GPTConfig
from model import TressaGPTModel
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
        streaming=True,
        token=os.environ.get("HF_TOKEN")
    )
    # Shuffle buffer
    hf_ds = hf_ds.shuffle(seed=42, buffer_size=config.shuffle_buffer_size)
    
    encoder = tiktoken.get_encoding("cl100k_base")
    
    # Set up Checkpointing State Tracking
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
    start_step = 0
    dataset_state = {"docs_consumed": 0}
    
    # 3. Initialize Model & Optimizer immediately to allow weights loading
    model = TressaGPTModel(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {total_params:,} trainable parameters.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    if os.path.exists(checkpoint_path):
        print(f"\n--- Resuming from Checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        dataset_state['docs_consumed'] = checkpoint.get('docs_consumed', 0)
        
        # Lossless fast-forwarding of the stream
        print(f"Fast-forwarding dataset by {dataset_state['docs_consumed']} documents...")
        hf_ds = hf_ds.skip(dataset_state['docs_consumed'])
    
    hf_iterator = iter(hf_ds)
    
    def create_dataloader(block_size, batch_size):
        stride = block_size // 2
        train_dataset = GPTStreamingDataset(
            hf_iterator=hf_iterator,
            encoder=encoder,
            block_size=block_size,
            stride=stride,
            state_dict=dataset_state
        )
        return DataLoader(train_dataset, batch_size=batch_size)
        
    # Find current curriculum block size based on the start_step
    current_block_size = config.curriculum_schedule[0]["block_size"]
    current_batch_size = config.curriculum_schedule[0]["batch_size"]
    for t_step in sorted(config.curriculum_schedule.keys()):
        if start_step >= t_step:
            current_block_size = config.curriculum_schedule[t_step]["block_size"]
            current_batch_size = config.curriculum_schedule[t_step]["batch_size"]
            
    train_loader = create_dataloader(current_block_size, current_batch_size)
    data_iter = iter(train_loader)
    
    # 4. Training Loop setup
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    model.train()
    
    print(f"\nStarting Training! Committing to 5 Billion Tokens ({config.max_steps} steps).")
    step = start_step
    start_time = time.time()
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
            if step == 0:
                print(f"Step {step:07d} | Loss: {loss.item():.4f}")
            else:
                elapsed = time.time() - start_time
                s_per_step = elapsed / 100
                eta_hrs = (config.max_steps - step) * s_per_step / 3600
                print(f"Step {step:07d} | Loss: {loss.item():.4f} | {s_per_step:.2f}s/step | ETA: {eta_hrs:.2f} hrs")
                start_time = time.time()
            
            
        step += 1
            
        # Periodic Checkpointing
        if step > start_step and step % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'docs_consumed': dataset_state['docs_consumed']
            }, ckpt_path)
            print(f"--- Auto-Checkpoint saved at Step {step} (Docs Consumed: {dataset_state['docs_consumed']}) ---")
            
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
