import os
import torch
import torch.nn.functional as F
import tiktoken
from config import GPTConfig
from model import TressaGPTModel # Targeting your new class name!

def generate(model, prompt, encoder, config, max_new_tokens=100, temperature=0.8, top_k=50):
    model.eval()
    
    # 1. Encode human readable prompt into Neural Tokens
    input_ids = encoder.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=config.device)
    
    # 2. Autoregressive Loop
    for _ in range(max_new_tokens):
        # Prevent the sequence from blowing past 1024 tokens and crashing positional embeddings
        x_cond = x if x.size(1) <= config.max_seq_len else x[:, -config.max_seq_len:]
        
        # Forward pass (no gradients needed for chat)
        with torch.no_grad():
            logits = model(x_cond)
            
        # specifically isolate the very last generated token
        logits = logits[:, -1, :] / temperature
        
        # Apply Top-K sampling to chop off bizarre hallucination branches
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Math conversion to percentages (Probabilities)
        probs = F.softmax(logits, dim=-1)
        
        # Sample entirely randomly from the top_k probabilities safely
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Decode the generated raw token string directly to console immediately
        token_str = encoder.decode([idx_next.item()])
        print(token_str, end="", flush=True)
        
        # Append the new token back into the continuous stream
        x = torch.cat((x, idx_next), dim=1)
        
    print("\n")

def main():
    config = GPTConfig()
    device = config.device
    print(f"Booting inference engine on {device}...")
    
    # 1. Initialize Blank Brain explicitly using your TressaGPTModel Class
    model = TressaGPTModel(config).to(device)
    
    # 2. Locate the 3.4B-Token Checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        return
        
    print(f"Loading weights from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_state_dict = checkpoint["model_state_dict"]
    
    # 3. THE MAGIC FIX: Stripping the compile prefix
    # Because we added `torch.compile()` to train.py, PyTorch secretly added "_orig_mod." 
    # to the front of every single neural layer name when it saved it to disk.
    # We strip it gracefully right here before loading!
    clean_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("_orig_mod."):
            clean_state_dict[key[10:]] = value
        else:
            clean_state_dict[key] = value
            
    # Load the perfectly clean dictionary!
    model.load_state_dict(clean_state_dict)
    
    docs_consumed = checkpoint.get("docs_consumed", 0)
    print(f"✅ Neural Network active! (Resumed from Step {checkpoint.get('step', '?')} - {docs_consumed} docs read)")
    
    # 4. Fire the Interactive Chat Generator
    encoder = tiktoken.get_encoding("cl100k_base")
    
    print("\n" + "="*50)
    print("Welcome to Tressa_GPT Interactive Neural Shell!")
    print("Type 'quit' or 'exit' to shut down.")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input("You >> ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                continue
                
            print("Tressa >> ", end="")
            generate(model, prompt, encoder, config, max_new_tokens=150)
        except (KeyboardInterrupt, EOFError):
            print("\nShutting down engine...")
            break

if __name__ == "__main__":
    main()
