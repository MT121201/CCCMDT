# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py

# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def modulate(x, shift, scale):
    """
    Modulates input `x` by applying a shift and scale.
    :param x: Input tensor.
    :param shift: Shift tensor.
    :param scale: Scale tensor.
    :return: Modulated tensor after applying `x * (1 + scale) + shift`.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    This helps in capturing the time information in embeddings.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # MLP network to process embeddings
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Creates sinusoidal timestep embeddings.
        :param t: (N,) Tensor containing timestep values.
        :param dim: Embedding dimension.
        :return: (N, D) Tensor with positional embeddings for time.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        """
        Processes timestep scalar into vector embeddings using MLP.
        :param t: (N,) Tensor of timesteps.
        :return: (N, hidden_size) Tensor of embedded timesteps.
        """
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Also includes label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)   
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels with a certain probability for classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels   

    def forward(self, labels, train, force_drop_ids=None):
        """
        Embeds labels, with optional dropout during training.
        :param labels: (N,) Tensor of class labels.
        :return: (N, hidden_size) Tensor of embedded class labels.
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    Defines a DiT transformer block with adaptive layer normalization conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        approx_GeLU = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_size, act_layer=approx_GeLU, drop=0.0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True)
        )

    def forward(self, x, c):
        """
        Forward pass with conditioning.
        :param x: (N, T, D) Input tensor.
        :param c: (N, D) Conditioning vector.
        :return: (N, T, D) Transformed tensor.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    Final layer that projects hidden states into output space.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2, bias=True)
        )
    
    def forward(self, x, c):
        """
        Final projection of hidden states.
        :param x: (N, T, D) Tensor of hidden states.
        :param c: (N, D) Conditioning vector.
        :return: (N, T, patch_size**2 * out_channels) Output tensor.
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion Transformer model for image generation.
    """
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=1000, learn_sigma=True):
        super().__init__()
        # Model components and initializations
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.x_embedder = PatchEmbed(patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # DiT blocks
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes model weights, including patch embeddings and DiT block normalization layers.
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def unpatchify(self, x):
        """
        Reshapes the output tensor from patch embeddings back to spatial format.
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, t, y):
        """
        Forward pass of DiT model.
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # Combine time and label embeddings
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, which also batches an unconditional forward pass for classifier-free guidance (CFG).
        :param x: (N, C, H, W) Tensor, input images.
        :param t: (N,) Tensor, timesteps.
        :param y: (N,) Tensor, class labels.
        :param cfg_scale: Scalar, guidance scale for classifier-free guidance.
        :return: (N, C, H, W) Tensor, model output with CFG applied.
        """
        # Split `x` into two halves, one for conditional, the other for unconditional guidance
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)  # Duplicate `half` for unconditional pass
        model_out = self.forward(combined, t, y)   # Run forward pass with combined input
        
        # Apply CFG to only the first three channels for reproducibility
        eps, rest = model_out[:, :3], model_out[:, 3:]  # Separate eps and additional channels
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)  # Split into conditional/unconditional parts

        # Apply CFG scaling to eps for controlled guidance
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)    # Concatenate back to original size

        # Concatenate modulated eps with rest of output channels
        return torch.cat([eps, rest], dim=1)



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Generates 2D sinusoidal positional embeddings.
    :param embed_dim: Dimension of embedding vectors.
    :param grid_size: Grid height and width.
    :param cls_token: Bool flag for including a class token.
    :param extra_tokens: Additional tokens (e.g., for special purposes).
    :return: 2D positional embeddings.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)        # Generate a 2D grid
    grid = np.stack(grid, axis=0)             # Stack along the first dimension
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed



def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generates positional embeddings based on a 2D grid.
    :param embed_dim: Embedding dimension.
    :param grid: 2D grid array of shape [2, 1, grid_size, grid_size].
    :return: 2D positional embeddings of shape [H*W, D].
    """
    assert embed_dim % 2 == 0

    # Split embedding space between height and width
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Computes sinusoidal positional embeddings for a sequence of positions. and one unconditional version.
    :param embed_dim: Embedding dimension.
    :param pos: Array of positions.
    :return: Sinusoidal embedding for each position.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # Flatten grid positions
    out = np.einsum('m,d->md', pos, omega)  # Outer product for sinusoidal pattern

    emb_sin = np.sin(out) # Sinusoidal part of embedding
    emb_cos = np.cos(out) # Cosine part of embedding

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}