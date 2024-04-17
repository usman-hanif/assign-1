from math import cos, pi

def learning_rate_schedule(t, min_lr, max_lr, num_warmup_iters, num_anneal_iters):
    if t <= num_warmup_iters:
        return t / num_warmup_iters * max_lr
    elif t <= num_anneal_iters and t > num_warmup_iters:
        return min_lr + 0.5 * (1 + cos(pi * (t - num_warmup_iters) / (num_warmup_iters - num_anneal_iters))) * (max_lr - min_lr)
    else:
        return min_lr