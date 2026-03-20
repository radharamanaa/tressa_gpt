from dataclasses import dataclass, field
import torch

@dataclass
class GPTConfig:
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
    max_seq_len: int = 1000
    curriculum_schedule: dict = field(default_factory=lambda: {
        0: {"block_size": 128, "batch_size": 32},
        800_000: {"block_size": 256, "batch_size": 16},
        1_000_000: {"block_size": 512, "batch_size": 8},
        1_150_000: {"block_size": 1000, "batch_size": 4}
    })
    
    # Dataset
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"
    
    # Path settings
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
