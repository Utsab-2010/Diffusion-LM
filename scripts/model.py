import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_latent, 3 * config.n_latent)
        self.c_proj = nn.Linear(config.n_latent, config.n_latent)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.n_latent = config.n_latent
        # Create a causal mask (lower triangular matrix) and register it as a buffer
        # A buffer is not a parameter, but is saved with the model state_dict
        self.register_buffer("bias", torch.tril(torch.ones(config.n_context, config.n_context))
                                     .view(1, 1, config.n_context, config.n_context))

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_latent, dim=2)
        
        # Reshape for multi-head attention: (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        
        # --- MASKING STARTS HERE ---
        # Apply the causal mask: fill "future" positions with -infinity
        # This makes their softmax probability zero.
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # --- MASKING ENDS HERE ---

        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, hs)
        
        # Re-assemble all head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        return y
    
class GPT2MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_latent, config.mlp_expansion*config.n_latent)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.mlp_expansion*config.n_latent, config.n_latent)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embed, config.n_latent)
        self.down_proj = nn.Linear(config.n_latent, config.n_embed)

        self.ln1 = nn.LayerNorm(config.n_latent,eps=1e-5,elementwise_affine=True)
        self.attn = GPT2Attention(config)
        self.ln2 = nn.LayerNorm(config.n_latent,eps=1e-5,elementwise_affine=True)
        self.mlp = GPT2MLP(config)

    def forward(self,x):

        h = self.up_proj(x)
        h = h + self.attn(self.ln1(h))
        h = h + self.mlp(self.ln2(h))
        
        return x + self.down_proj(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) #
        # TODO: Double check the ordering here
        return embeddings
    
class LMEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.n_vocab,config.n_embed)
    
    def forward(self,input_ids):
        x = self.embed(input_ids)
        
        return x
        
class Denoiser(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Embedding(config.n_vocab,config.n_embed),
            wpe = nn.Embedding(config.n_context,config.n_embed),
            drop = nn.Dropout(0.1,inplace=False),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embed,eps=1e-5,elementwise_affine=True)
        ))
        
        # self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

        self.small_mlp = nn.Linear(config.n_embed, config.n_embed)

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(config.n_embed),
            nn.Linear(config.n_embed, config.n_embed),
            nn.GELU()
            )

    def forward(self,input_embeddings,time_step, targets=None):
        B,T,C = input_embeddings.size()
        device = input_embeddings.device

        pos = torch.arange(0,T,dtype=torch.long,device=device).unsqueeze(0)  # (1,T)
        x = input_embeddings +  self.transformer.wpe(pos)  # (B,T,C) pytorch does braodcasting for the position embeddingss and adds them to the token embeddings 
        
        time_emb = self.time_embed(time_step) # (B, C)
        x= x + time_emb.unsqueeze(1)  # (B, T, C)
        
        x = self.transformer.drop(x)


        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # (B,T,C)
        # logits = self.lm_head(x)  # (B,T,vocab_size) 
        # we don't need the head since we are not doing autoregressive language modeling
        
        # we want to predict the starting sequence before the noising part.
        x = self.small_mlp(x)  # (B,T,C)
        
        return x


class Decoding(nn.Module):
    def __init__(self,config):
        super().__init__()
    # takes x0 (B,T,C) and give a softmax over vocab size           
        self.l1 = nn.Linear(config.n_embed, config.n_vocab, bias=False)
        
        
    def forward(self,x):
        x = self.l1(x)
        # x = F.softmax(x,dim=-1)

        return x

class DiffusionLM(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embedding = LMEmbedding(config)
        self.denoiser = Denoiser(config)
        self.decoder = Decoding(config)
        
    def forward(self,input_ids,time_step, targets=None):
        input_embeddings = self.embedding(input_ids)  # (B,T,C)
        x = self.denoiser(input_embeddings,time_step, targets)  # (B,T,C)
        logits = x@self.embedding.embed.weight.T  # (B,T,vocab_size)
        
        return x, logits