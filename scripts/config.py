from dataclasses import dataclass

@dataclass
class gpt2config:
    n_vocab: int = 50257
    n_layer: int = 12
    n_embed: int = 128
    n_context: int = 1024
    n_head: int = 8
    n_timesteps: int = 1000
    mlp_expansion: int = 4
    n_latent: int = 768


