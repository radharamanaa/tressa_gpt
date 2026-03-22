import os
import sys
from dotenv import load_dotenv

# Explicitly load local .env variables so users don't have to 'export'
load_dotenv()

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from config import GPTConfig

def verify_token_permissions(api: HfApi, repo_id: str):
    """
    Test Case: Checks if the provided token has permissions to write to the repository.
    """
    print(f"Testing token permissions for write access to '{repo_id}'...")
    try:
        # whoami() validates if the token is broadly valid
        user_info = api.whoami()
        username = user_info.get("name", "Unknown User")
        print(f"Authenticated successfully as Hub User: {username}")
        
        # Test write permission by attempting to access/create the target repo
        api.create_repo(repo_id, exist_ok=True)
        print("✅ Permission Verified: Token has write access!")
        return True
        
    except HfHubHTTPError as e:
        status_code = getattr(e.response, "status_code", None)
        if status_code in [401, 403]:
            print(f"❌ Permission Denied: The HF_TOKEN provided does not have write access.")
            print("Please ensure your token is a 'Write' token generated in your Hugging Face account settings.")
        else:
            print(f"❌ HF API Error during verification: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unknown error during token verification: {e}")
        return False

def push_model_to_hf(repo_id: str, checkpoint_path: str):
    """
    Pushes the trained model checkpoint to Hugging Face Hub.
    Requires HF_TOKEN environment variable to be set.
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please run the training loop (`python src/train.py`) before pushing.")
        sys.exit(1)
        
    # Strictly pull token from env variables
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN could not be found.")
        print("Please ensure it is set inside your local '.env' file: `HF_TOKEN=your_token`")
        sys.exit(1)
        
    # Initialize API strictly using the environment token
    api = HfApi(token=token)
    
    # Run the test case to guarantee permissions before pushing big files
    if not verify_token_permissions(api, repo_id):
        sys.exit(1)

    # Upload the .pt weights file
    print(f"\nUploading '{checkpoint_path}'...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="tressa_gpt_step840k_3.4B_tokens.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    
    # Upload the config file for transparent reproducibility
    config_path = "src/config.py" # Assumes script is run from project root
    if os.path.exists(config_path):
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.py",
            repo_id=repo_id,
            repo_type="model",
        )
        print("Uploaded architecture config parameters successfully.")

    print(f"\n🎉 Model completely pushed! View your model here: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change 'your-username/my-first-gpt-5B' to your actual HF repo name
    USER_REPO = "abhijeetmishra101/tressa_gpt_50M" 
    
    config = GPTConfig()
    CHECKPOINT = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
    
    choice = input(f"Do you want to run the push pipeline for '{USER_REPO}'? (y/n): ")
    if choice.lower() == 'y':
        push_model_to_hf(USER_REPO, CHECKPOINT)
    else:
        print("Pipeline aborted.")
