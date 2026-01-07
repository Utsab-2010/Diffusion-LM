#!/usr/bin/env python3
"""
Diffusion Language Model - Inference Script
Real-time text generation with clean terminal visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math
import os
import sys
import time
from dataclasses import dataclass

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.model import DiffusionLM
from scripts.config import gpt2config


# ============================================================================
# ANSI COLORS & TERMINAL CONTROL
# ============================================================================

class Colors:
    """Clean color palette - Mac inspired"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Main colors
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Background
    BG_DARK = '\033[48;5;234m'
    BG_LIGHT = '\033[48;5;236m'


def clear_lines(n):
    """Clear n lines above cursor"""
    for _ in range(n):
        sys.stdout.write('\033[F')  # Move cursor up
        sys.stdout.write('\033[K')  # Clear line


def print_header():
    """Print clean header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.RESET}        {Colors.BOLD}Diffusion Language Model - Inference{Colors.RESET}         {Colors.CYAN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚═══════════════════════════════════════════════════════════╝{Colors.RESET}\n")


def print_progress_bar(current, total, bar_length=40):
    """Print a clean progress bar"""
    progress = current / total
    filled = int(bar_length * progress)
    bar = '━' * filled + '─' * (bar_length - filled)
    percentage = progress * 100
    
    return f"{Colors.CYAN}[{Colors.GREEN}{bar}{Colors.CYAN}]{Colors.RESET} {percentage:5.1f}%"


# ============================================================================
# TOKENIZER
# ============================================================================

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def finalize_tokens(x0_final, embedding_weights):
    """
    Converts the final denoised latent into discrete token IDs.
    Args:
        x0_final: Tensor of shape (B, T, C)
        embedding_weights: Tensor of shape (Vocab, C)
    """
    distances = torch.cdist(x0_final, embedding_weights.unsqueeze(0), p=2)  # (B,T,Vocab)  
    token_ids = torch.argmin(distances, dim=-1)  # (B, T)
    return token_ids


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"{Colors.GRAY}Loading checkpoint from: {checkpoint_path}{Colors.RESET}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        config = checkpoint['config']
        alpha_bars = checkpoint['alpha_bars']
        T = checkpoint['T']
        
        model = DiffusionLM(config).to(device)
        
        # Handle compiled model state_dict (with _orig_mod. prefix)
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print(f"{Colors.GRAY}  Detected compiled model, removing '_orig_mod.' prefix...{Colors.RESET}")
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"{Colors.GREEN}✓ Model loaded successfully{Colors.RESET}")
        print(f"{Colors.GRAY}  Config: {config.n_layer} layers, {config.n_embed} embed dim, {config.n_head} heads{Colors.RESET}")
        print(f"{Colors.GRAY}  Timesteps: T={T}{Colors.RESET}\n")
        
        return model, config, alpha_bars, T
    except FileNotFoundError:
        print(f"{Colors.YELLOW}Error: Checkpoint file not found{Colors.RESET}")
        sys.exit(1)
    except KeyError as e:
        print(f"{Colors.YELLOW}Error: Missing checkpoint key{Colors.RESET}")
        sys.exit(1)
    except RuntimeError:
        print(f"{Colors.YELLOW}Error: Model state_dict mismatch{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.YELLOW}Error: {type(e).__name__}{Colors.RESET}")
        sys.exit(1)


# ============================================================================
# INFERENCE WITH VISUALIZATION
# ============================================================================

def reverse_diffusion_with_viz(model, alpha_bars, T, tokenizer, context_length=64, 
                                batch_size=1, clamping_start=0.4, skip_step=1):
    """
    Reverse diffusion process with real-time terminal visualization
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Start from pure noise
    x_t = torch.randn(batch_size, context_length, model.config.n_embed, device=device)
    
    # Calculate total steps
    total_steps = sum(1 for t in range(T, 0, -1) if t % skip_step == 0 or t == T)
    current_step = 0
    
    print(f"{Colors.BOLD}Starting reverse diffusion...{Colors.RESET}\n")
    
    # Reserve space for the display (we'll update in place)
    display_lines = 6
    for _ in range(display_lines):
        print()
    
    with torch.no_grad():
        for t_step in range(T, 0, -1):
            # Skip steps based on skip_step parameter
            if t_step % skip_step == 0 or t_step == T:
                pass
            else:
                continue
            
            current_step += 1
            t_tensor = torch.tensor([t_step] * batch_size, device=device)
            
            # Predict x0 from x_t
            x0_pred = model.denoiser(x_t, t_tensor)
            
            # Apply clamping in later timesteps
            if t_step < clamping_start * T:
                x0_clamped_tokens = finalize_tokens(x0_pred, model.embedding.embed.weight)
                x0_clamped = model.embedding(x0_clamped_tokens)
            else:
                x0_clamped = x0_pred
            
            # Sample noise for next step
            epsilon = torch.randn_like(x_t)
            
            # Compute x_{t-1}
            if t_step > 1:
                x_t = torch.sqrt(alpha_bars[t_step - 1]) * x0_clamped + \
                      torch.sqrt(1 - alpha_bars[t_step - 1]) * epsilon
            else:
                x_t = x0_clamped
            
            # Convert to tokens
            generated_tokens = finalize_tokens(x0_clamped, model.embedding.embed.weight)
            generated_text = tokenizer.decode(generated_tokens[0].tolist())
            generated_text_clean = tokenizer.clean_text(generated_text)
            
            # Calculate progress
            progress = current_step / total_steps
            
            # Move cursor up and clear previous display
            clear_lines(display_lines)
            
            # Display current state
            print(f"{Colors.BOLD}Timestep:{Colors.RESET} {Colors.YELLOW}{t_step:4d}{Colors.RESET}/{T}  │  "
                  f"{Colors.BOLD}Step:{Colors.RESET} {Colors.CYAN}{current_step:4d}{Colors.RESET}/{total_steps}  │  "
                  f"{Colors.BOLD}Phase:{Colors.RESET} {Colors.MAGENTA}{'Clamping' if t_step < clamping_start * T else 'Refining'}{Colors.RESET}")
            print(print_progress_bar(current_step, total_steps))
            print()
            print(f"{Colors.BOLD}{Colors.BLUE}Generated Text:{Colors.RESET}")
            print(f"{Colors.WHITE}{Colors.BG_DARK} {generated_text_clean[:100]:<100} {Colors.RESET}")
            print()
            
            # Small delay for visual effect (remove for max speed)
            time.sleep(0.01)
    
    # Generate final output from x_t (AFTER the loop completes)
    generated_tokens = finalize_tokens(x_t, model.embedding.embed.weight)
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    generated_text_clean = tokenizer.clean_text(generated_text)
    
    # Final output
    clear_lines(display_lines)
    print(f"{Colors.GREEN}{Colors.BOLD}✓ Generation Complete!{Colors.RESET}\n")
    print(f"{Colors.BOLD}Final Output:{Colors.RESET}")
    print(f"{Colors.WHITE}{Colors.BG_DARK} {generated_text_clean} {Colors.RESET}\n")
    
    return generated_tokens, generated_text_clean


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Configuration
    CHECKPOINT_PATH = "saved_models/checkpoints_E2E_v1/diff_lm_checkpoint.pt"
    CONTEXT_LENGTH = 64
    BATCH_SIZE = 1
    CLAMPING_START = 0.5
    SKIP_STEP = 5  # Sample every N timesteps for faster inference
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print_header()
    print(f"{Colors.BOLD}Device:{Colors.RESET} {Colors.CYAN}{device.upper()}{Colors.RESET}")
    print(f"{Colors.BOLD}Precision:{Colors.RESET} {Colors.CYAN}{'HIGH' if device == 'cuda' else 'STANDARD'}{Colors.RESET}\n")
    
    # Load model
    model, config, alpha_bars, T = load_model(CHECKPOINT_PATH, device)
    
    # Initialize tokenizer
    tokenizer = MyTokenizer(max_len=CONTEXT_LENGTH)
    
    # Run inference
    print(f"{Colors.BOLD}Inference Parameters:{Colors.RESET}")
    print(f"  {Colors.GRAY}Sequence Length: {CONTEXT_LENGTH}{Colors.RESET}")
    print(f"  {Colors.GRAY}Clamping Start: {CLAMPING_START * 100:.0f}%{Colors.RESET}")
    print(f"  {Colors.GRAY}Skip Step: {SKIP_STEP} (sampling every {SKIP_STEP} timesteps){Colors.RESET}\n")
    
    generated_tokens, generated_text = reverse_diffusion_with_viz(
        model=model,
        alpha_bars=alpha_bars,
        T=T,
        tokenizer=tokenizer,
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
        clamping_start=CLAMPING_START,
        skip_step=SKIP_STEP
    )
    
    print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.DIM}Inference complete. Generated {len(generated_text.split())} words.{Colors.RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠ Interrupted by user{Colors.RESET}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.YELLOW}Error: {type(e).__name__}{Colors.RESET}\n")
        sys.exit(1)
