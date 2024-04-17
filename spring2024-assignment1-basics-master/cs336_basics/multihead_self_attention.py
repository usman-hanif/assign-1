import torch
import torch.nn as nn
from scaled_dot_product_attention import scaled_dot_product_attention

class MultiHead(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop):
        super(MultiHead, self).__init__()
        self.d_model = d_model 
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_k = d_model // num_heads
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x): 
        seq_length = x.size(dim=1)

        device = x.device

        q = self.Wq(x)
        
        
        k, v = self.Wk(x), self.Wv(x)

        

        # print("This is the shape of WqX", q.shape)
        # print("This is the shape of WkX in forward", k.shape)
        # print("This is the shape of WvX in forward", v.shape)


        q = q.view(-1, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
        k = k.view(-1, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
        v = v.view(-1, seq_length, self.num_heads, self.d_k).transpose(1, 2) 





        # print("These are the shapes, post view changing")

        # print("This is the shape of WqX reshaped: ", q.shape)
        # print("This is the shape of WkX reshaped: ", k.shape)
        # print("This is the shape of WvX reshaped: ", v.shape)

        
        mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.bool), diagonal=1)

        # print(mask)
        # print(mask.shape)

        result = scaled_dot_product_attention(q, k, v, mask, self.attn_pdrop)


        result = result.transpose(1, 2)
        

        result = result.reshape(-1, seq_length, self.d_model)


        result = self.Wo(result)


        # print("This is the shape of the result we get: ", result.shape)

        return result

    



