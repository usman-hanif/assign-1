import torch 

def cross_entropy_loss(logits, targets):

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits = logits - logits_max
    
   
    exp_logits = torch.exp(logits)
    sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_softmax = logits - torch.log(sum_exp_logits)
    
    
    targets = targets.unsqueeze(-1)
    true_class_log_probs = torch.gather(log_softmax, dim=-1, index=targets).squeeze(-1)
    
   
    loss = -true_class_log_probs.mean()
    
    return loss












