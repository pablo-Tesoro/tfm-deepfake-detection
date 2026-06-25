"""
Explicabilidad espacial mediante Grad-CAM.

Grad-CAM genera un mapa de calor que indica qué regiones de la imagen han pesado
más en la decisión del modelo. Se calcula a partir de los gradientes de la
puntuación de "fake" respecto a las activaciones de la última capa convolucional
del backbone.

Pieza clave del diseño: reutilizamos la cabeza del *baseline* ya entrenado (que
clasifica a partir del embedding de un frame) montada sobre el backbone para
formar un clasificador POR FOTOGRAMA. Así explicamos la componente espacial sin
entrenar nada nuevo. La decisión temporal (LSTM) se explica aparte, con la
probabilidad por frame a lo largo del clip (ver notebook 03).

Detalle técnico: como el backbone está congelado, forzamos `requires_grad` en la
entrada para que el gradiente fluya hasta sus capas convolucionales.

Requiere PyTorch (+ timm) y OpenCV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class FrameScorer(nn.Module):
    """Backbone + cabeza por frame -> logit de 'fake' para una imagen de rostro."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)      # [B, D]
        return self.head(feats)       # [B, 1]


def get_target_layer(backbone: nn.Module) -> nn.Module:
    """Devuelve la capa convolucional objetivo para Grad-CAM.

    Cubre EfficientNet (conv_head) y ResNet (layer4); en otro caso, usa la última
    capa Conv2d encontrada.
    """
    if hasattr(backbone, "conv_head"):
        return backbone.conv_head
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]
    convs = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No se encontró ninguna capa convolucional en el backbone.")
    return convs[-1]


class GradCAM:
    """Grad-CAM sobre una capa convolucional objetivo."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Args:
            input_tensor: [B, C, H, W] imágenes de rostro normalizadas.

        Returns:
            Mapas de calor [B, h, w] normalizados a [0, 1].
        """
        # Forzar gradiente en la entrada: necesario porque el backbone está
        # congelado y, si no, el gradiente no llegaría a sus capas convolucionales.
        input_tensor = input_tensor.clone().requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(input_tensor)          # [B, 1]
        score = logits[:, 0].sum()
        score.backward()

        # Pesos = media espacial de los gradientes (importancia de cada canal).
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)       # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1)                  # [B, h, w]
        cam = torch.relu(cam)

        # Normalizar cada mapa a [0, 1].
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.cpu().numpy()


def overlay_cam_on_image(
    rgb_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Superpone el mapa de calor sobre la imagen original.

    Args:
        rgb_image: imagen RGB uint8 [H, W, 3].
        cam: mapa de calor [h, w] en [0, 1] (se redimensiona a la imagen).
        alpha: peso del mapa de calor en la mezcla.

    Returns:
        Imagen RGB uint8 con el mapa superpuesto.
    """
    import cv2

    h, w = rgb_image.shape[:2]
    cam_resized = cv2.resize(cam.astype("float32"), (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype("uint8"), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * rgb_image).astype("uint8")
    return overlay


def load_frame_scorer(
    backbone_name: str,
    baseline_checkpoint: str | Path,
    device: Optional[str] = None,
):
    """Construye el FrameScorer: backbone preentrenado + cabeza del baseline entrenado.

    Args:
        backbone_name: modelo timm (debe coincidir con el usado en embeddings).
        baseline_checkpoint: ruta a baseline.pt (cabeza entrenada en la Fase 2).
        device: 'cuda'/'cpu'. None autodetecta.

    Returns:
        (scorer, transform, target_layer, device).
    """
    import timm
    from src.models.baseline import FrameMeanBaseline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    embed_dim = backbone.num_features

    state = torch.load(baseline_checkpoint, map_location=device)
    hidden = state["classifier.0.weight"].shape[0]      # inferir tamaño oculto
    baseline = FrameMeanBaseline(embed_dim, hidden=hidden)
    baseline.load_state_dict(state)

    scorer = FrameScorer(backbone, baseline.classifier).to(device).eval()

    data_cfg = timm.data.resolve_data_config({}, model=backbone)
    transform = timm.data.create_transform(**data_cfg)
    target_layer = get_target_layer(backbone)
    return scorer, transform, target_layer, device
