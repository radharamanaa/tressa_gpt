import math
import torch
import torch.nn as nn

class _SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len, :]

class _SingleHeadCausalAttention(nn.Module):
    def __init__(self, head_dim: int, embed_dim: int, drop_out: float = 0.2):
        super().__init__()
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(in_features=embed_dim, out_features=head_dim)
        self.Wk = nn.Linear(in_features=embed_dim, out_features=head_dim)
        self.Wv = nn.Linear(in_features=embed_dim, out_features=head_dim)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, inputs):
        query_t = self.Wq(inputs)
        key_t = self.Wk(inputs)
        val_t = self.Wv(inputs)
        
        attention_score = query_t @ key_t.transpose(1, 2)
        
        # Causal masking (upper triangular is forced to -inf)
        T = attention_score.shape[-1]
        mask = torch.triu(torch.ones(T, T, device=inputs.device), diagonal=1).bool()
        attention_weights = attention_score.masked_fill(mask, float('-inf'))
        
        att_weights = torch.softmax(attention_weights / self.head_dim ** 0.5, dim=-1)
        att_weights_dropped = self.drop_out(att_weights)
        return att_weights_dropped @ val_t

class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, no_of_att_heads: int, drop_out: float = 0.2):
        super().__init__()
        dim_per_head = embed_dim // no_of_att_heads
        self.modulesList = nn.ModuleList([
            _SingleHeadCausalAttention(head_dim=dim_per_head, embed_dim=embed_dim, drop_out=drop_out) 
            for _ in range(no_of_att_heads)
        ])
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, inputs):
        multi_head_out = [head(inputs) for head in self.modulesList]
        concat_out = torch.cat(multi_head_out, dim=-1)
        return self.out_proj(concat_out)

class _TransformerBlock(nn.Module):
    def __init__(self, no_of_heads: int, embed_dim: int, ffn_expansion: int, drop_out: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.layer_n_before = nn.LayerNorm(embed_dim)
        self.drop_out1 = nn.Dropout(drop_out)
        self.drop_out2 = nn.Dropout(drop_out)
        self.layer_n_after = nn.LayerNorm(embed_dim)
        
        self.multi_head_att = _MultiHeadAttention(embed_dim, no_of_heads, drop_out)
        
        # Feed Forward Network setup with expansion
        self.feed_for_netw = nn.Sequential(
            nn.Linear(embed_dim, ffn_expansion * embed_dim), 
            nn.GELU(),
            nn.Linear(ffn_expansion * embed_dim, embed_dim)
        )

    def forward(self, inputs):
        context_vector = self.drop_out1(self.multi_head_att(self.layer_n_before(inputs)))
        x = context_vector + inputs
        
        norm = self.layer_n_after(x)
        ffn_output = self.feed_for_netw(norm)
        ffn_output = self.drop_out2(ffn_output)
        
        return ffn_output + x

class GPTStyleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.token_embed = nn.Embedding(config.vocab_size, embedding_dim=config.embed_dim)
        self.pos_embed = _SinusoidalPositionalEmbedding(d_model=config.embed_dim, max_len=config.max_seq_len)
        self.embed_dropout = nn.Dropout(config.drop_out)
        
        self.transformers = nn.ModuleList([
            _TransformerBlock(config.no_of_heads, config.embed_dim, config.ffn_expansion, config.drop_out) 
            for _ in range(config.no_of_trans_blocks)
        ])
        
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Tie weights to save parameters
        self.lm_head.weight = self.token_embed.weight

    def forward(self, input_token_ids):
        token_embed = self.token_embed(input_token_ids)
        seq_len = input_token_ids.shape[-1]
        pos_embed = self.pos_embed(seq_len).to(input_token_ids.device)
        
        data = self.embed_dropout(token_embed + pos_embed)
        
        for trans in self.transformers:
            data = trans(data)

        # LayerNorm occurs before final projection
        data = self.layer_norm(data)
        return self.lm_head(data)
