from transformer_block import TransformerBlock
from torch import nn
from rmsnorm import RMSNorm
import torch

class TransformerLM(nn.Module):
    def __init__(self,vocab_size: int,context_length: int,d_model: int,num_layers: int,num_heads: int,d_ff: int,attn_pdrop: float,residual_pdrop: float):
        super(TransformerLM, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(residual_pdrop)
    
    def forward(self, x):
        x = self.token_embeddings(x)
        x = x + self.position_embeddings(torch.arange(x.size(1), device=x.device))
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x