import torch
from torch import nn


class PatchEncoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, patch_size: int) -> None:
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.flatten = nn.Flatten(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2d(x)
        out = self.flatten(out)

        return out.permute(0, 2, 1)


class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, dropout_p: float) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.msa = nn.MultiheadAttention(embedding_dim, n_heads, dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer_norm(x)
        out, _ = self.msa(query=out, key=out, value=out, need_weights=False)

        return out


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_size: int, dropout_p: float) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer_norm(x)
        out = self.mlp(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(
            self, 
            embedding_dim: int,
            n_heads: int,
            mlp_size: int,
            mlp_dropout: float,
            attn_dropout: float,
    ) -> None:
        super().__init__()

        self.attn_block = AttentionBlock(embedding_dim, n_heads, attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn_block(x) + x
        out = self.mlp_block(out) + out

        return out
    

class ViT(nn.Module):
    model_name: str = 'ViT'
    
    def __init__(
            self,
            n_channels: int,
            img_size: int,
            patch_size: int,
            n_transformer_layers: int,
            embedding_dim: int,
            mlp_size: int,
            n_heads: int,
            attn_dropout: float,
            mlp_dropout: float,
            emb_dropout: float,
            n_classes: int,
    ) -> None:
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.class_embedding = nn.Parameter(
            torch.rand((1, 1, embedding_dim)),
            requires_grad=True,
        )

        self.positional_embedding = nn.Parameter(
            torch.rand(1, self.num_patches + 1, embedding_dim),
            requires_grad=True,
        )

        self.embedding_dropout = nn.Dropout(emb_dropout)

        self.patch_embedding_block = PatchEncoder(n_channels, embedding_dim, patch_size)

        self.transformers_blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim, n_heads, mlp_size, mlp_dropout, attn_dropout)
            for _ in range(n_transformer_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        class_tocken = self.class_embedding.expand(batch_size, -1, -1)
        
        embedded_patch = self.patch_embedding_block(x)
        patch_and_class = torch.cat((class_tocken, embedded_patch), dim=1)
        
        res = patch_and_class + self.positional_embedding
        res_sparse = self.embedding_dropout(res)

        transformers_out = self.transformers_blocks(res_sparse)
        prediction = self.classifier(transformers_out[:, 0])

        return prediction.squeeze()
