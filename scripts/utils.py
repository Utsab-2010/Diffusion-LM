import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import tiktoken
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from collections import Counter
import subprocess
import spacy

from .model import DiffusionLM


class MyTokenizer:
    def __init__(self, max_len):
        tokenizer = tiktoken.get_encoding("r50k_base")
        self.special_tokens = {
            "<pad>": tokenizer.n_vocab,
            "<bos>": tokenizer.n_vocab + 1,
            "<eos>": tokenizer.n_vocab + 2,
        }
        self.tokenizer = tiktoken.Encoding(
            name="r50k_base_ext",
            pat_str=tokenizer._pat_str,
            mergeable_ranks=tokenizer._mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.n_vocab = self.tokenizer.n_vocab 
        self.max_len = max_len

    @staticmethod
    def clean_text(text):
        for tok in ("<pad>", "<bos>", "<eos>"):
            text = text.replace(tok, "")
        return text
    
    def encode(self, text, max_len=None):
        if max_len is None:
            max_len = self.max_len
        ids = self.tokenizer.encode(text, allowed_special=set())
        ids = [self.special_tokens["<bos>"]] + ids + [self.special_tokens["<eos>"]]

        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = self.special_tokens["<eos>"]
        else:
            ids += [self.special_tokens["<pad>"]] * (max_len - len(ids))
        return ids  

    def decode(self, ids):
        return self.tokenizer.decode(ids)


def get_next_log_filename(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    existing_logs = [f for f in os.listdir(log_dir) if f.startswith('log_') and f.endswith('.txt')]
    
    if not existing_logs:
        return os.path.join(log_dir, 'log_1.txt')
    
    log_numbers = []
    for log_file in existing_logs:
        try:
            num = int(log_file.replace('log_', '').replace('.txt', ''))
            log_numbers.append(num)
        except ValueError:
            continue
    
    next_num = max(log_numbers) + 1 if log_numbers else 1
    return os.path.join(log_dir, f'log_{next_num}.txt')


def save_checkpoint(model, config, alpha_bars, T, checkpoint_name, save_individual=True):
    checkpoint_dir = f'saved_models/checkpoints_{checkpoint_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'config': config,
        'model_state_dict': model.state_dict(),
        'alpha_bars': alpha_bars,
        'T': T
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, 'diff_lm_checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    if save_individual:
        model_path = os.path.join(checkpoint_dir, 'model.pt')
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model state dict saved to {model_path}")
    
    print(f"\nCheckpoint summary:")
    print(f"  Directory: {checkpoint_dir}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"  Timesteps (T): {T}")
    print(f"  Vocab size: {config.n_vocab}")


def load_checkpoint(checkpoint_name, device='cuda', eval_mode=True):
    checkpoint_dir = f'saved_models/checkpoints_{checkpoint_name}'
    checkpoint_path = os.path.join(checkpoint_dir, 'diff_lm_checkpoint.pt')
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    loaded_config = checkpoint['config']
    loaded_alpha_bars = checkpoint['alpha_bars'].to(device)
    loaded_T = checkpoint['T']
    
    model = DiffusionLM(loaded_config).to(device)
    
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("Detected compiled model, removing '_orig_mod.' prefix...")
        state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    if eval_mode:
        model.eval()
    
    sqrt_ab = torch.sqrt(loaded_alpha_bars)
    sqrt_1mab = torch.sqrt(1 - loaded_alpha_bars)
    
    print(f"\n✓ Model loaded successfully!")
    print(f"  Config: n_vocab={loaded_config.n_vocab}, n_layer={loaded_config.n_layer}, n_embed={loaded_config.n_embed}")
    print(f"  T={loaded_T}, Alpha bars range: [{loaded_alpha_bars.min():.4f}, {loaded_alpha_bars.max():.4f}]")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    return model, loaded_config, loaded_alpha_bars, loaded_T, sqrt_ab, sqrt_1mab


def posterior_mean(x_t, x0, t, alpha_bars, alphas):
    abar_t = alpha_bars[t]
    abar_tm1 = alpha_bars[t-1] if t > 0 else torch.tensor(1.0, device=x_t.device)
    
    eps = 1e-8
    coef1 = torch.sqrt(abar_tm1 + eps) * (1 - alphas[t]) / (1 - abar_t + eps)
    coef2 = torch.sqrt(alphas[t] + eps) * (1 - abar_tm1) / (1 - abar_t + eps)
    
    return coef1 * x0 + coef2 * x_t


def rounding_weight(it, max_iters, round_start=None, round_warmup=None, round_max_weight=0.4):
    if round_start is None:
        round_start = int(0.2 * max_iters)
    if round_warmup is None:
        round_warmup = int(0.5 * max_iters)
    
    return 1
    k = 10 / round_warmup 
    x0 = round_start + (round_warmup / 2)
    weight = round_max_weight / (1 + math.exp(-k * (it - x0)))
    return weight if it >= round_start else 0.0


def get_batch(split, batch_size, sequence_length, train_encoded, test_encoded, device):
    data_encoded = train_encoded if split == 'train' else test_encoded
    indices = torch.randint(0, len(data_encoded), (batch_size,))
    w_stack = data_encoded[indices].to(device)
    return w_stack


def finalize_tokens(x0_final, embedding_weights):
    distances = torch.cdist(x0_final, embedding_weights.unsqueeze(0), p=2)
    token_ids = torch.argmin(distances, dim=-1)
    return token_ids


def reverse_diffusion_with_clamping(model, config, tokenizer, alpha_bars, T, context_length=50, 
                                    batch_size=1, clamping_start=0.4, skip_step=1, display_at_steps=None, device='cuda'):
    model.eval()
    
    x_t = torch.randn(batch_size, context_length, config.n_embed, device=device)
    
    if display_at_steps is None:
        display_at_steps = [1]
    
    print(f"\n{'='*70}")
    print(f"Starting Reverse Diffusion")
    print(f"{'='*70}")
    print(f"Total Timesteps: {T} | Context Length: {context_length}")
    print(f"Clamping Start: {clamping_start*100:.0f}% | Skip Step: {skip_step}")
    print(f"{'='*70}\n")
    
    print(f"🌀 Initial State (t={T}, Pure Noise):")
    initial_tokens = finalize_tokens(x_t, model.embedding.embed.weight)
    initial_text = tokenizer.decode(initial_tokens[0].tolist())
    initial_text_clean = tokenizer.clean_text(initial_text)
    print(f"{initial_text_clean}")
    print(f"{'-'*70}\n")
    
    with torch.no_grad():
        for t_step in range(T, 0, -1):
            if t_step % skip_step == 0 or t_step == T:
                pass
            else:
                continue
            
            t_tensor = torch.tensor([t_step] * batch_size, device=device)
            x0_pred = model.denoiser(x_t, t_tensor)
            
            if t_step < clamping_start * T:
                x0_clamped_tokens = finalize_tokens(x0_pred, model.embedding.embed.weight)
                x0_clamped = model.embedding(x0_clamped_tokens)
            else:
                x0_clamped = x0_pred
            
            epsilon = torch.randn_like(x_t)
            
            if t_step > 1:
                x_t = torch.sqrt(alpha_bars[t_step - 1]) * x0_clamped + \
                      torch.sqrt(1 - alpha_bars[t_step - 1]) * epsilon
            else:
                x_t = x0_clamped
            
            if t_step in display_at_steps and t_step != T:
                generated_tokens = finalize_tokens(x0_clamped, model.embedding.embed.weight)
                generated_text = tokenizer.decode(generated_tokens[0].tolist())
                generated_text_clean = tokenizer.clean_text(generated_text)
                
                phase = "🔒 Clamping" if t_step < clamping_start * T else "✨ Refining"
                print(f"{phase} Intermediate State (t={t_step}):")
                print(f"{generated_text_clean}")
    
    generated_tokens = finalize_tokens(x_t, model.embedding.embed.weight)
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    generated_text_clean = tokenizer.clean_text(generated_text)
    
    print(f"\n{'='*70}")
    print(f"\nFinal Output:")
    print(f"{generated_text_clean}")
    print(f"\n{'='*70}\n")
    
    return generated_tokens, generated_text_clean


@torch.no_grad()
def visualize_embeddings_2d(emb_func, vocab_list, top_n=5000):
    embeddings = emb_func.weight[:top_n].detach().cpu().float().numpy()
    
    print("Pre-reducing dimensions with PCA...")
    pca = PCA(n_components=8)
    embeddings_reduced = pca.fit_transform(embeddings)

    print(f"Running 2D t-SNE on {top_n} tokens...")
    tsne = TSNE(n_components=2, perplexity=10, init='pca', verbose=1, random_state=42)
    embeds_2d = tsne.fit_transform(embeddings_reduced)

    print("Performing POS tagging...")
    try:
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model... (one-time setup)")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
        
        pos_tags = []
        for token_text in vocab_list[:top_n]:
            clean_token = token_text.strip()
            if clean_token:
                doc = nlp(clean_token)
                pos = doc[0].pos_ if len(doc) > 0 else "OTHER"
            else:
                pos = "OTHER"
            pos_tags.append(pos)
    except Exception as e:
        print(f"Warning: Could not perform POS tagging ({e}), using default colors")
        pos_tags = ["OTHER"] * top_n
    
    unique_pos = sorted(set(pos_tags))
    pos_to_color = {pos: i for i, pos in enumerate(unique_pos)}
    colors = [pos_to_color[pos] for pos in pos_tags]
    
    cmap = plt.cm.get_cmap('tab20', len(unique_pos))
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], 
                         alpha=0.6, s=8, c=colors, cmap=cmap)

    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=cmap(pos_to_color[pos]), 
                         markersize=8, label=pos) 
              for pos in unique_pos]
    plt.legend(handles=handles, title="Part of Speech", 
              loc='center left', bbox_to_anchor=(1, 0.5), 
              frameon=True, fontsize=9)

    plt.title(f"Diffusion-LM Latent Space (Top {top_n} Tokens) - Colored by POS")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    pos_counts = Counter(pos_tags)
    print("\nPOS Tag Distribution:")
    for pos, count in pos_counts.most_common():
        print(f"  {pos:12s}: {count:5d} ({100*count/top_n:.1f}%)")


def fwd_diffusion(x0, t, alpha_bars):
    a = alpha_bars[t].view(-1, 1, 1).to(x0.device)
    noise = torch.randn_like(x0)
    xt = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise
    return xt
