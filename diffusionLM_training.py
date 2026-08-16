# import sys
# sys.path.append('/media/linux-stuff/gpt2-diff/scripts')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch import device
from torch.optim.lr_scheduler import LambdaLR

import math
import os
import pandas as pd

from scripts.config import gpt2config
from scripts.model import DiffusionLM, LMEmbedding, Denoiser, Decoding
from scripts.utils import (
    MyTokenizer, 
    get_next_log_filename, 
    save_checkpoint, 
    load_checkpoint,
    posterior_mean,
    rounding_weight,
    get_batch,
    finalize_tokens,
    reverse_diffusion_with_clamping,
    visualize_embeddings_2d,
    infer_test_infilling,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
torch.set_float32_matmul_precision('medium')


tokenizer = MyTokenizer(max_len=13)
# tokenizer.decode(tokenizer.encode("Hello, tiktoken is fast!"))
config = gpt2config(n_vocab=tokenizer.n_vocab,n_layer=12,n_embed=16,n_head= 12, mlp_expansion=4,n_latent=12*32)
model = DiffusionLM(config).to(device)
print(f"Total Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(config.n_vocab)

import pandas as pd

# Load E2E dataset - extract text from 'ref' column
df = pd.read_csv('datasets/e2e-dataset/trainset.csv')
text = ' '.join(df['ref'].tolist())

print(f"Dataset length: {len(text)} characters")
print(f"Number of samples: {len(df)}")
print(f"First sample: {df['ref'][0]}")


# Split into train and test
train_size = int(0.9 * len(df))
train_df = df[:train_size].reset_index(drop=True)
test_df = df[train_size:].reset_index(drop=True)

print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# Pre-encode all sequences for training efficiency
print("\nEncoding training data...")
train_encoded = []
for idx, row in train_df.iterrows():
    encoded = tokenizer.encode(row['ref'], max_len=64)  # Use fixed sequence length
    train_encoded.append(encoded)
    if (idx + 1) % 5000 == 0:
        print(f"Encoded {idx + 1}/{len(train_df)} train samples")

print("\nEncoding test data...")
test_encoded = []
for idx, row in test_df.iterrows():
    encoded = tokenizer.encode(row['ref'], max_len=64)
    test_encoded.append(encoded)

# Convert to tensors
train_encoded = torch.tensor(train_encoded, dtype=torch.long)
test_encoded = torch.tensor(test_encoded, dtype=torch.long)

print(f"\nTrain encoded shape: {train_encoded.shape}")
print(f"Test encoded shape: {test_encoded.shape}")



# Training configuration — matched to Diffusion-LM paper Appendix B
# Paper: AdamW, lr=1e-4 linear decay, batch=64, 200K iters for E2E
max_iters = 250000          # paper: 200K for E2E, 800K for ROCStories
learning_rate = 2.5e-5        # paper Appendix B: "linearly decay learning rate starting at 1e-4"
eval_iters = 250            # log every N iterations
eval_batches = 10           # number of batches to average for eval loss
batch_size = 16             # paper Appendix B: batch size of 64
sequence_length = 64
T = 1000                    # paper §6.1 + Appendix A: 2000 diffusion steps
num_timestep_samples = 50    # sample this many timesteps per iteration
weight_decay = 1e-5         # AdamW default; paper doesn't specify exact value

# Sqrt noise schedule (paper Appendix A): ᾱ_t = 1 − √(t/T + s), s=1e-4
s = 1e-4
t = torch.arange(0, T+1, device=device, dtype=torch.float32)
alpha_bars = 1 - torch.sqrt(t / T + s)
alpha_bars = torch.clamp(alpha_bars, min=0.001, max=0.999)
alphas = torch.zeros(T+1, device=device)  # α_0 … α_T
alphas[0] = alpha_bars[0]
alphas[1:] = alpha_bars[1:] / alpha_bars[:-1]
alphas = torch.clamp(alphas, min=0.001, max=0.999)

# Precompute sqrt terms for efficiency
sqrt_ab = torch.sqrt(alpha_bars)
sqrt_1mab = torch.sqrt(1 - alpha_bars)

print(f"Alpha bars range: [{alpha_bars.min():.4f}, {alpha_bars.max():.4f}]")
print(f"Alphas range: [{alphas.min():.4f}, {alphas.max():.4f}]")
print(f"Initial noise std (√(1−ᾱ₁)): {torch.sqrt(1 - alpha_bars[1]):.4f}  — paper says ~0.1")


if __name__ == "__main__":

    model = torch.compile(model, mode='max-autotune')  # or 'max-autotune' for more optimization
    optimizer_model = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_lambda = lambda step: 1.0 - (step / float(max_iters))
    scheduler_model = LambdaLR(optimizer_model, lr_lambda=lr_lambda)


    log_file = get_next_log_filename('logs')
    print(f"Logging to: {log_file}")

    with open(log_file, 'w') as f:
        f.write("Iteration,Total_Loss,Denoising_Loss,Anchor_Loss,Rounding_Loss,Val_Loss\n")

    checkpoint_counter = 0

    for it in range(0, max_iters):

        # ---- forward diffusion: w → x0 = EMB(w) + σ₀·ε  (paper eq. 2) ----
        w = get_batch('train', batch_size, sequence_length, train_encoded=train_encoded, test_encoded=test_encoded, device=device)
        w_emb = model.embedding(w)
        x0 = w_emb + 0.1 * torch.randn_like(w_emb)   # σ₀ = 0.1
        total_loss = 0.0

        # denoising_loss = denoising_loss / num_timestep_samples   # ← average over samples
        K = num_timestep_samples
        B = batch_size

        # expand x0 to (K*B, T, C) — K copies of the same x0
        x0_expanded = x0.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, sequence_length, x0.size(-1))

        # sample K*B independent noises and timesteps
        eps_all = torch.randn_like(x0_expanded)
        t_all = torch.randint(1, T + 1, (K * B,), device=device)

        # noise the copies
        sqrt_ab_all = sqrt_ab[t_all].view(K * B, 1, 1)
        sqrt_1mab_all = sqrt_1mab[t_all].view(K * B, 1, 1)
        xt_all = sqrt_ab_all * x0_expanded + sqrt_1mab_all * eps_all

        # single forward pass through the denoiser
        x0_hat_all = model.denoiser(xt_all, t_all)

        # MSE against the expanded x0
        denoising_loss = F.mse_loss(x0_hat_all, x0_expanded)
        total_loss += denoising_loss

        # ---- 2) anchor loss:  ‖EMB(w) − fθ(x₁, 1)‖²  (paper eq. 2, second term) ----
        xt_1 = sqrt_ab[1] * x0 + sqrt_1mab[1] * torch.randn_like(x0)
        x0_hat_1 = model.denoiser(xt_1, torch.ones(batch_size, device=device))
        anchor_loss = F.mse_loss(x0_hat_1, w_emb)
        total_loss += anchor_loss

        # ---- 3) rounding loss:  −log pθ(w | x̂₀)  (paper eq. 2, third term) ----
        # logits = x0_hat_1 @ model.embedding.embed.weight.T
        logits = model.decoder(x0_hat_1)
        rounding_loss = F.cross_entropy(logits.view(-1, config.n_vocab), w.view(-1))
        total_loss += rounding_loss

        # ---- NaN / Inf guard ----
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n{'='*70}")
            print(f"TRAINING STOPPED: NaN/Inf detected at iteration {it}")
            print(f"{'='*70}")
            print(f"Loss Diagnostics:")
            print(f"  Total Loss:     {total_loss.item() if not torch.isnan(total_loss) else 'NaN'}")
            print(f"  Denoising:      {denoising_loss.item()}")
            print(f"  Anchor:         {anchor_loss.item()}")
            print(f"  Rounding:       {rounding_loss.item()}")
            print(f"\nModel Output Statistics:")
            # print(f"  x0_hat range:   [{x0_hat.min().item():.2f}, {x0_hat.max().item():.2f}]")
            print(f"  logits range:   [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            print(f"\nGradient Statistics:")
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"  Total grad norm: {total_norm:.4f}")
            print(f"{'='*70}\n")
            break

        optimizer_model.zero_grad(set_to_none=True)
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer_model.step()
        scheduler_model.step()

        # ---- logging ----
        with open(log_file, 'a') as f:
            f.write(f"{it},{total_loss.item():.6f},{denoising_loss.item():.6f},{anchor_loss.item():.6f},{rounding_loss.item():.6f},0.0\n")

        if it % 1 == 0:
            print(f"Iter {it:5d} | lr {scheduler_model.get_last_lr()[0]:.2e} | "
                f"loss {total_loss.item():.4f} | denoise {denoising_loss.item():.4f} | "
                f"anchor {anchor_loss.item():.4f} | round {rounding_loss.item():.4f}", end='\r')

        # ---- periodic validation (every 1250 iters) ----
        if it % (eval_iters * 5) == 0 and it > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(eval_batches):
                    w_val = get_batch('test', batch_size, sequence_length, train_encoded=train_encoded, test_encoded=test_encoded, device=device)
                    w_val_emb = model.embedding(w_val)
                    x0_val = w_val_emb + 0.1 * torch.randn_like(w_val_emb)
                    t_val = torch.randint(1, T + 1, (batch_size,), device=device)
                    xt_val = sqrt_ab[t_val].view(batch_size, 1, 1) * x0_val + sqrt_1mab[t_val].view(batch_size, 1, 1) * torch.randn_like(x0_val)
                    x0_hat_val = model.denoiser(xt_val, t_val)
                    val_loss += F.mse_loss(x0_hat_val, x0_val).item()
            val_loss /= eval_batches
            # model.train()
            print(f"         >>> val mse = {val_loss:.4f}")

            # unconditional generation test
            context_length = 64
            generated_tokens, generated_text = reverse_diffusion_with_clamping(
                model=model,
                config=config,
                tokenizer=tokenizer,
                alpha_bars=alpha_bars,
                T=T,
                context_length=context_length,
                batch_size=1,
                clamping_start=1.0,
                skip_step=50,
                display_at_steps=[T//2, 1],
                device=device
            )
            print("Generated Text:", generated_text)
            model.train()

        # ---- checkpoint (every 20000 iters, paper: 200K total) ----
        if it % 5000 == 0 and it > 0:
            checkpoint_name = f"training_ckpt_{checkpoint_counter % 2}"
            save_checkpoint(model, config, alpha_bars, T, checkpoint_name, save_individual=False)
            checkpoint_counter += 1

    print(f"\nTraining complete! Logs saved to: {log_file}")