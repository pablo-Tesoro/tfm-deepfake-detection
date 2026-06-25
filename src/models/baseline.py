"""
Baseline sin modelado temporal.

Promedia los embeddings de los frames de un vídeo (ignorando el padding) y los
clasifica con un MLP. Es la referencia que la arquitectura híbrida debe superar:
si añadir la dimensión temporal (LSTM) no mejora sobre promediar frames, hay que
discutir por qué. Cumple el requisito de comparar varias técnicas.

Requiere PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrameMeanBaseline(nn.Module):
    """Pooling por media de embeddings + MLP -> logit de 'fake'."""

    def __init__(self, embed_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] secuencias de embeddings (con padding).
            lengths: [B] longitud real de cada secuencia.

        Returns:
            Logits [B] (sin sigmoide).
        """
        # Media solo sobre los frames válidos (excluye el padding).
        mask = (torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None])
        mask = mask.unsqueeze(-1).float()                      # [B, T, 1]
        summed = (x * mask).sum(dim=1)                          # [B, D]
        pooled = summed / lengths.clamp(min=1).unsqueeze(-1)    # media
        return self.classifier(pooled).squeeze(-1)
