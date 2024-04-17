import torch 
import torch.nn as nn

from bpe_working import train_bpe
from rmsnorm import RMSNorm
from gelu import gelu
from positionwise_ff import my_ffn
from softmax import softmax
from scaled_dot_product_attention import scaled_dot_product_attention
from multihead_self_attention import MultiHead

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop):
        super(TransformerBlock, self).__init__()

        self.rms_norm1 = RMSNorm(d_model)

        self.mhsa = MultiHead(d_model, num_heads, attn_pdrop)

        self.dropout1 = nn.Dropout(residual_pdrop) if residual_pdrop is not None else nn.Identity()


        self.rms_norm2 = RMSNorm(d_model)

        self.my_ffn = my_ffn(d_model, d_ff)

        self.dropout2 = nn.Dropout(residual_pdrop) if residual_pdrop is not None else nn.Identity()


    def forward(self, x): 
        normalized_x = self.rms_norm1(x)
        attn_out = self.mhsa(normalized_x)
        x = x + self.dropout1(attn_out)

        normalized_x_2 = self.rms_norm2(x)
        ffn_out = self.my_ffn(normalized_x_2)
        x = x + self.dropout2(ffn_out)

        return x










