import os
import torch
import torch.nn.functional as F
import tiktoken
from src.config import GPTConfig
from src.model import TressaGPTModel

def generate_text(model, encoder, prompt, max_new_tokens=50, device="cpu"):
    """
    Autoregressive generation loop taking the context and predicting the next token.
    """
    model.eval()
    context = torch.tensor(encoder.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # Disable autograd for massively faster generation
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Ensure context doesn't violently exceed the model's block size limit
            idx_cond = context[:, -model.pos_embed.pe.shape[0]:]
            
            # Forward pass
            logits = model(idx_cond)
            
            # Pluck out the logits for the final time step
            logits = logits[:, -1, :]
            
            # Apply softmax to map to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution (adds temperature randomness basically)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append isolated token back to sequence
            context = torch.cat((context, idx_next), dim=1)
            
            # Model mathematically terminates thought
            if idx_next.item() == encoder.eot_token:
                break
                
    generated_text = encoder.decode(context[0].tolist())
    return generated_text

def main():
    config = GPTConfig()
    device = config.device
    
    print("Loading cl100k_base tokenizer...")
    encoder = tiktoken.get_encoding("cl100k_base")
    
    print(f"Instantiating model on {device}...")
    model = TressaGPTModel(config).to(device)
    
    checkpoint_path = os.path.join(config.checkpoint_dir, "gpt_model_5B_tokens.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Successfully injected trained parameters!")
    else:
        print("⚠️ Warning: No trained checkpoint found. Operating with randomly initialized weights!")
        print(f"   (Failed to find checkpoint at: {checkpoint_path})")

    print("\n" + "="*50)
    print("🤖 First GPT Model Interactive Console")
    print("Type 'exit' to quit the console.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower().strip() == 'exit':
                print("Ending Session.")
                break
            
            if not user_input.strip():
                continue
                
            response = generate_text(model, encoder, user_input, max_new_tokens=100, device=device)
            # Removes the original prompt from the printed output for cleaner chatting
            clean_response = response[len(user_input):] 
            print(f"\nModel: {clean_response}\n")
            
        except KeyboardInterrupt:
            print("\nEnding Session.")
            break

if __name__ == "__main__":
    main()
