import os
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, TaskType

load_dotenv()
from config import GPTConfig
from model import TressaGPTModel

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} || All Parameters: {all_params:,} || Tuning %: {100 * trainable_params / all_params:.2f}%")

class InstructionDataset(Dataset):
    def __init__(self, hf_ds, encoder, max_len=512):
        self.data = []
        print("Formatting dataset and applying -100 Attention Masks...")
        for row in hf_ds:
            ans = row.get("output", "")
            
            # Format Prompt
            instr = row.get("instruction", "")
            inp = row.get("input", "")
            if inp:
                prompt = f"Instruction: {instr}\nInput: {inp}\nResponse: "
            else:
                prompt = f"Instruction: {instr}\nResponse: "
                
            prompt_tokens = encoder.encode(prompt)
            ans_tokens = encoder.encode(ans) + [encoder.eot_token]
            
            full_tokens = prompt_tokens + ans_tokens
            # Truncate if too long
            if len(full_tokens) > max_len:
                full_tokens = full_tokens[:max_len]
                
            # Create Targets: (-100 is ignored by PyTorch CrossEntropyLoss natively!)
            # We ONLY want loss calculated over the answer tokens.
            labels = [-100] * len(prompt_tokens) + ans_tokens
            if len(labels) > max_len:
                labels = labels[:max_len]
                
            # Pad sequences so they stack into tensors perfectly
            pad_len = max_len - len(full_tokens)
            if pad_len > 0:
                full_tokens.extend([encoder.eot_token] * pad_len)
                labels.extend([-100] * pad_len) # Pad tokens get -100 mask too
                
            self.data.append((torch.tensor(full_tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))

    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx): 
        return self.data[idx]

def main():
    config = GPTConfig()
    
    # 1. Load Base Model
    model = TressaGPTModel(config).to(config.device)
    checkpoint_path = os.path.join(config.checkpoint_dir, "tressa_gpt_50M.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading Pre-Trained Weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: Pre-trained checkpoint not found. Starting from scratch!")
        
    # 2. Inject LoRA into Custom Layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["Wq", "Wk", "Wv", "out_proj"], # Connecting exactly to your scratch architecture
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    # 3. Load & Format Alpaca Dataset
    hf_ds = load_dataset("tatsu-lab/alpaca", split="train")
    encoder = tiktoken.get_encoding("cl100k_base")
    train_dataset = InstructionDataset(hf_ds, encoder, max_len=1024)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # 4. Optimizer (Only affects LoRA injected parameters!)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # SFT learning rates are smaller
    
    # 5. Training Loop
    print("\n--- Starting Stage 1: Instruction Fine-Tuning ---")
    model.train()
    
    epochs = 2 # SFT is usually 1-3 epochs
    for epoch in range(epochs):
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)
            
            logits = model(x_batch) 
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y_batch[..., 1:].contiguous()
            
            # PyTorch intrinsically ignores the prompt & pad because we mapped them to -100
            loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Epoch {epoch} | Step {step:04d} | SFT Loss: {loss.item():.4f}")
                
    # Save the specialized Stage 1 adapter model
    stage1_dir = os.path.join(config.checkpoint_dir, "tressa_gpt_instruct_lora")
    model.save_pretrained(stage1_dir)
    print(f"\n✅ Stage 1 LoRA adapters securely saved to {stage1_dir}")

if __name__ == "__main__":
    main()
