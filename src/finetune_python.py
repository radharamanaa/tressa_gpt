import os
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from peft import PeftModel

load_dotenv()
from config import GPTConfig
from model import TressaGPTModel
from finetune_instruct import InstructionDataset # Reusing your formatter!

def main():
    config = GPTConfig()
    
    # 1. Load Base Pretrained Brain
    base_model = TressaGPTModel(config).to(config.device)
    checkpoint_path = os.path.join(config.checkpoint_dir, "tressa_gpt_50M.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
    # 2. Re-attach the Instruction-Tuning LoRA weights from Stage 1!
    stage1_dir = os.path.join(config.checkpoint_dir, "tressa_gpt_instruct_lora")
    if os.path.exists(stage1_dir):
        print(f"Loading Stage 1 Instructor LoRA from {stage1_dir}...")
        model = PeftModel.from_pretrained(base_model, stage1_dir)
        print("Merging Stage 1 adapters permanently into the base model brain...")
        base_model = model.merge_and_unload()
    else:
        print("Warning: Stage 1 LoRA not found. Ensure you run finetune_instruct.py first.")
        return
        
    # 3. Inject a BRAND NEW LoRA for Stage 2 Python Specialization!
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["Wq", "Wk", "Wv", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} || All Parameters: {all_params:,} || Tuning %: {100 * trainable_params / all_params:.2f}%")
        
    # 3. Load & Format the 18k Python Execution Dataset
    print("\nLoading Python 18k Execution Dataset...")
    hf_ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    encoder = tiktoken.get_encoding("cl100k_base")
    train_dataset = InstructionDataset(hf_ds, encoder, max_len=1024)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) # Smaller learning rate for Stage 2
    
    # 5. Training Loop
    print("\n--- Starting Stage 2: Python Code Fine-Tuning ---")
    model.train()
    
    epochs = 2
    for epoch in range(epochs):
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)
            
            logits = model(x_batch) 
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y_batch[..., 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Epoch {epoch} | Step {step:04d} | Python Coding Loss: {loss.item():.4f}")
            
    # Save Final Python Adapter
    stage2_dir = os.path.join(config.checkpoint_dir, "tressa_gpt_python_lora")
    model.save_pretrained(stage2_dir)
    print(f"\n✅ Stage 2 Python LoRA efficiently saved to {stage2_dir}!")

if __name__ == "__main__":
    main()
