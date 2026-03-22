from dataclasses import dataclass, field
import torch

@dataclass
class GPTConfig:
    # Hugging Face PEFT Boilerplate Requirement
    model_type: str = "tressa_gpt"
    
    # Model parameters
    embed_dim: int = 384
    no_of_trans_blocks: int = 6
    no_of_heads: int = 6
    vocab_size: int = 100277
    drop_out: float = 0.1
    
    # FFN Expansion
    ffn_expansion: int = 5 

    # Training logic
    batch_size: int = 32
    block_size: int = 128
    learning_rate: float = 3e-4
    max_steps: int = 1_220_703  # 5,000,000,000 tokens / (batch_size * block_size)
    
    # Curriculum Learning mapping training steps to (block_size, batch_size)
    max_seq_len: int = 1024
    curriculum_schedule: dict = field(default_factory=lambda: {
        # Forced strictly to maximum context length for the remaining 380,000 steps!
        0: {"block_size": 1024, "batch_size": 4} 
    })
    
    # Fault Tolerance
    checkpoint_interval: int = 5000
    shuffle_buffer_size: int = 10000
    
    # Dataset
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"
    
    # Path settings
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
