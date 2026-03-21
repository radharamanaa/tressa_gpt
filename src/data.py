import torch
from torch.utils.data import IterableDataset

class GPTStreamingDataset(IterableDataset):
    def __init__(self, hf_iterator, encoder, block_size, stride, state_dict=None):
        self.hf_iterator = hf_iterator
        self.encoder = encoder
        self.block_size = block_size
        self.stride = stride
        self.state_dict = state_dict if state_dict is not None else {"docs_consumed": 0}

    def __iter__(self):
        # We process the streaming HuggingFace iterator lazily
        for doc in self.hf_iterator:
            self.state_dict["docs_consumed"] += 1
            # Filtering rules based on FineWeb-Edu quality scores
            if doc.get("language_score", 1.0) < 0.7 or doc.get("score", 5) < 3:
                continue
                
            tokens = self.encoder.encode(doc["text"])
            tokens.append(self.encoder.eot_token)
            
            # We need at least enough tokens for one block + 1 (for the target shift)
            if len(tokens) < self.block_size + 1:
                continue
                
            # Slide over the tokens
            for i in range(0, len(tokens) - self.block_size, self.stride):
                # Standard causal modeling: predict the next token continuously
                x = tokens[i : i + self.block_size]
                y = tokens[i + 1 : i + self.block_size + 1]
                
                # Yield instances as PyTorch Tensors
                yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
