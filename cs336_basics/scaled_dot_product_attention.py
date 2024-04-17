import torch
from softmax import softmax
import math
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask, pdrop): 
    ## Ensure Q and K match (batch_size, ..., seq_len, d_k)
    assert Q.size(0) == K.size(0)
    assert Q.size(-1) == K.size(-1)
    assert Q.size(-2) == K.size(-2)

    # print("Shape of Q inside scaled dot product:", Q.shape)
    # print("Shape of K_T inside scaled dot product", K.transpose(-1, -2).shape)



    Q_K = Q @ K.transpose(-1, -2) 


    # print("Shape of Q_KT:", Q_K.shape)

    # Get d_k and square root it
    d_k = K.size(-1)
    # print("This is d_k shape inside scaled: ", d_k)
    denom = math.sqrt(d_k)
    # print("This is after squaring it inside scaled: ", denom)

    # Inner val for softmax 
    scores = Q_K / denom

    # print("This is the inner value that we will now softmax:", scores)
    # print("This is the inner value shape that we will be softmaxing", scores.shape)
  

    ## If there exists a mask, see what needs to be masked to zero 
    if mask is not None: 
        if scores.dim() == 4: 
          
            # mask = mask.unsqueeze(0).unsqueeze(1)  
            # mask = mask.expand(scores.size(0), scores.size(1), -1, -1)  
            scores[:, :, mask] = float('-inf')
            # print("This is the mask shape", mask.shape)
            # print("This is the scores shape", scores.shape)


        elif scores.dim() == 3: 
            # mask = mask.unsqueeze(0) 
            # mask = mask.expand(scores.size(0), -1, -1)  
            # print("This is the mask shape", mask.shape)
            # print("This is the scores shape", scores.shape)
            scores[:, mask] = float('-inf')

      
        # scores = scores.masked_fill(mask, float('-inf'))




    
    result = softmax(scores, -1)

    if pdrop is not None: 
        result = F.dropout(result, pdrop)


    return result @ V






