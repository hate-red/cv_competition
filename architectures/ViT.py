import torch
from torch import Tensor
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            patch_size: int, 
            n_channels: int, 
            embedding_dim: int
        ):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Conv2d(n_channels, embedding_dim, patch_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, seq_len: int):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.attn(x, x, x)[0]
    

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int, attn_dropout: float):
        super().__init__()

        self.attn = MultiHeadAttention(embedding_dim, num_heads, attn_dropout)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embedding_dim)
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x

class ViT(nn.Module):
    model_name: str = 'ViT'
    
    def __init__(
            self, 
            img_size: int, 
            n_channels: int,
            patch_size: int,
            embedding_dim: int, 
            num_heads: int, 
            num_layers: int, 
            mlp_dim: int,
            attn_dropout: float,
            num_classes: int,
        ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_size, n_channels, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, (img_size // patch_size) ** 2)
        
        self.transformer_blocks = nn.Sequential(*[
            TransformerEncoderBlock(embedding_dim, num_heads, mlp_dim, attn_dropout) 
            for _ in range(num_layers)
        ])

        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        B = x.size(0)
        x = self.patch_embedding(x)

        cls_tokens = self.class_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        
        x = self.transformer_blocks(x)
        
        return self.classifier(x[:, 0])