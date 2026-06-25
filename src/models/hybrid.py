"""
Arquitectura híbrida espacio-temporal.

CNNLSTM: el modelo principal. Sobre las secuencias de embeddings (ya extraídos
por la CNN congelada) aplica una RNN (LSTM o GRU) que evalúa la COHERENCIA
TEMPORAL del clip (parpadeos, micromovimientos) y clasifica real/fake.

EndToEndModel: variante que une el backbone CNN y la cabeza temporal en un solo
modelo, con opción de congelar/descongelar el backbone. Permite el experimento
de FINE-TUNING (descongelar las últimas capas) frente al modo feature-extraction.
Su entrada son secuencias de IMÁGENES de rostros, no embeddings; se usará cuando
se quiera comparar fine-tuning vs. feature-extraction.

Requiere PyTorch (+ timm para EndToEndModel).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """RNN temporal sobre secuencias de embeddings -> logit de 'fake'."""

    def __init__(
        self,
        embed_dim: int,
        hidden: int = 256,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] secuencias de embeddings (con padding).
            lengths: [B] longitud real de cada secuencia.

        Returns:
            Logits [B].
        """
        # Empaquetar para que la RNN ignore el padding.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden_state = self.rnn(packed)
        h_n = hidden_state[0] if self.rnn_type == "lstm" else hidden_state
        # h_n: [num_layers * num_dir, B, hidden]. Tomamos la última capa.
        if self.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # concatena ambas direcciones
        else:
            feat = h_n[-1]
        return self.head(feat).squeeze(-1)


class EndToEndModel(nn.Module):
    """Backbone CNN (timm) + cabeza temporal, con congelado opcional.

    Entrada: [B, T, C, H, W] (secuencia de imágenes de rostros).
    Permite comparar fine-tuning (freeze_backbone=False) vs. feature-extraction
    (freeze_backbone=True). Más costoso que CNNLSTM porque procesa imágenes.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        hidden: int = 256,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.set_backbone_trainable(not freeze_backbone)
        embed_dim = self.backbone.num_features
        self.temporal = CNNLSTM(
            embed_dim, hidden=hidden, num_layers=num_layers,
            rnn_type=rnn_type, dropout=dropout,
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Congela o descongela el backbone (para alternar feature-ext/fine-tuning)."""
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)
        self.backbone.train(trainable)

    def forward(self, frames: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W] secuencias de imágenes de rostros.
            lengths: [B] longitudes reales.
        """
        b, t, c, h, w = frames.shape
        flat = frames.view(b * t, c, h, w)
        feats = self.backbone(flat)            # [B*T, D]
        seq = feats.view(b, t, -1)             # [B, T, D]
        return self.temporal(seq, lengths)
