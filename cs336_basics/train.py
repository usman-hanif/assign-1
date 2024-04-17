import torch
from src.transformer_lm import TransformerLM
from argparse import ArgumentParser
import numpy as np
from src.data_loader import load_data
from src.cross_entropy import cross_entropy_loss
# from torch.optim import AdamW
from src.adamW import AdamW
from src.learning_rate_schedule import learning_rate_schedule
from tqdm import tqdm
from gradient_clipping import gradient_clipping
import wandb
import time
    
def update_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available"
    # wandb.login()
    # run = wandb.init(project="cs336_a1", config={"vocab_size": args.vocab_size, "context_length": args.context_length, "d_model": args.d_model, "num_layers": args.num_layers, "num_heads": args.num_heads, "d_ff": args.d_ff, "attn_pdrop": args.attn_pdrop, "residual_pdrop": args.residual_pdrop, "batch_size": args.batch_size, "num_steps": args.num_steps, "min_lr": args.min_lr, "max_lr": args.max_lr, "num_warmup_iters": args.num_warmup_iters, "num_anneal_iters": args.num_anneal_iters, "adamw_beta1": args.adamw_beta1, "adamw_beta2": args.adamw_beta2, "adamw_epsilon": args.adamw_epsilon, "gradient_clip_val": args.gradient_clip_val})
    device = args.device
    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.attn_pdrop,
        args.residual_pdrop,
    ).to(device)
    model.train()
    params = model.parameters()
    optimizer = AdamW(params=params, lr=args.max_lr, betas=(args.adamw_beta1, args.adamw_beta2), eps=args.adamw_epsilon)
    total_loss = 0
    
    train_token_ids = np.memmap(args.train_data_path, dtype=np.int16, mode='r')
    val_token_ids = np.memmap(args.val_data_path, dtype=np.int16, mode='r')
    
    for i in tqdm(range(args.num_steps), total=args.num_steps):
        lr = learning_rate_schedule(i, args.min_lr, args.max_lr, args.num_warmup_iters, args.num_anneal_iters)
        update_optimizer_lr(optimizer, lr)
        # start = time.time()
        seq_in, target = load_data(train_token_ids, args.batch_size, args.context_length, device)
        # end = time.time()
        # print(f"Data loading took {end - start} seconds")
        # start = time.time()
        logits = model(seq_in)
        # end = time.time()
        # print(f"Forward pass took {end - start} seconds")
        # start = time.time()
        loss = cross_entropy_loss(logits, target)
        # end = time.time()
        # print(f"Loss calculation took {end - start} seconds")
        # print(i, loss.item())
        # start = time.time()
        gradient_clipping(params, args.gradient_clip_val)
        # end = time.time()
        # print(f"Gradient clipping took {end - start} seconds")
        # start = time.time()
        loss.backward()
        # end = time.time()
        # print(f"Backward pass took {end - start} seconds")
        # start = time.time()
        optimizer.step()
        # end = time.time()
        # print(f"Optimizer step took {end - start} seconds")
        optimizer.zero_grad()
        
        # Logging
        total_loss += loss.item()
        if i % args.log_interval == 0:
            print(f"Step {i}: Loss = {total_loss / args.log_interval}")
            # wandb.log({"loss": total_loss / args.log_interval})
            total_loss = 0
        
        # if i % args.eval_interval == 0:
        #     model.eval()
        #     val_loss = 0
        #     with torch.no_grad():
        #         for i in range(args.num_steps_val):
        #             val_seq_in, val_target = load_data(val_token_ids, args.batch_size_val, args.context_length, device)
        #             val_logits = model(val_seq_in)
        #             val_loss = cross_entropy_loss(val_logits, val_target)
        #             val_loss += val_loss.item()
        #         print(f"Validation Loss = {val_loss / args.batch_size_val}")
        #         wandb.log({"val_loss": val_loss / args.batch_size_val})
        #     model.train()

        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="results/tokenizer/ids_tinystories_train_int16.npy", help="Path to the whole batch sequence .npy or .bin file")
    parser.add_argument("--val_data_path", type=str, default="results/tokenizer/ids_tinystories_valid_int16.npy", help="Path to the validation sequence .npy or .bin file")
    parser.add_argument('--vocab_size', type=int, default=10000, help="Size of the vocabulary")
    parser.add_argument('--context_length', type=int, default=256, help="Length of the context window")
    parser.add_argument('--d_model', type=int, default=512, help="Dimensionality of the hidden states")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers")
    parser.add_argument('--num_heads', type=int, default=16, help="Number of attention heads")
    parser.add_argument('--d_ff', type=int, default=2048, help="Dimensionality of the feedforward layer")
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help="Dropout probability for attention")
    parser.add_argument('--residual_pdrop', type=float, default=0.1, help="Dropout probability for residual connections")
    parser.add_argument("--train_data", type=str, default="", help="Path to the training data")
    parser.add_argument("--val_data", type=str, default="", help="Path to the validation data")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Path to save the model")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=2500, help="Number of steps to train the model")
    parser.add_argument('--batch_size_val', type=int, default=32, help="Batch size for validation")
    parser.add_argument("--num_steps_val", type=int, default=1000, help="Number of steps to evaluate the model on the validation set")
    parser.add_argument("--min_lr", type=float, default=0, help="Minimum learning rate")
    parser.add_argument("--max_lr", type=float, default=1e-2, help="Maximum learning rate")
    parser.add_argument("--num_warmup_iters", type=int, default=10000, help="Number of warmup iterations")
    parser.add_argument("--num_anneal_iters", type=int, default=70000, help="Number of annealing iterations")
    parser.add_argument("--adamw_beta1", type=float, default=0.9, help="Beta1 for the AdamW optimizer")
    parser.add_argument("--adamw_beta2", type=float, default=0.999, help="Beta2 for the AdamW optimizer")
    parser.add_argument("--adamw_epsilon", type=float, default=1e-8, help="Epsilon for the AdamW optimizer")
    parser.add_argument("--adamw_weight_decay", type=float, default=0.01, help="Weight decay for the AdamW optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Value to clip the gradients to")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--log_interval", type=int, default=100, help="Interval to log the training loss")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Interval to evaluate the model on the validation set")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Path to the checkpoint to resume training from")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train the model on")
    args = parser.parse_args()
    main(args)