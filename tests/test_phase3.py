"""Pruebas de la Fase 3 (Grad-CAM). Requiere torch + timm + opencv."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import timm

from src.models.baseline import FrameMeanBaseline
from src.explainability.gradcam import (
    FrameScorer, GradCAM, get_target_layer, overlay_cam_on_image,
)


def _build_scorer(freeze_backbone=True, hidden=32):
    # pretrained=False evita descargar pesos en el entorno de pruebas
    backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)
    embed_dim = backbone.num_features
    head = FrameMeanBaseline(embed_dim, hidden=hidden).classifier
    return FrameScorer(backbone, head).eval(), backbone


def test_target_layer():
    _, backbone = _build_scorer()
    layer = get_target_layer(backbone)
    assert isinstance(layer, torch.nn.Conv2d), type(layer)
    print("  [OK] get_target_layer (capa conv encontrada)")


def test_gradcam_with_frozen_backbone():
    # El caso real: backbone CONGELADO. Verifica que el gradiente llega igualmente.
    scorer, backbone = _build_scorer(freeze_backbone=True)
    target = get_target_layer(backbone)
    cam = GradCAM(scorer, target)

    img = torch.randn(2, 3, 224, 224)
    heat = cam(img)

    assert cam.activations is not None, "No se capturaron activaciones"
    assert cam.gradients is not None, "No se capturaron gradientes (backbone congelado)"
    assert heat.shape[0] == 2 and heat.ndim == 3, f"forma inesperada {heat.shape}"
    assert heat.min() >= 0.0 and heat.max() <= 1.0 + 1e-6, "fuera de [0,1]"
    assert heat.max() > 0.0, "mapa todo a cero"
    print(f"  [OK] GradCAM con backbone congelado (mapa {heat.shape[1]}x{heat.shape[2]})")


def test_overlay():
    rgb = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    cam = np.random.rand(7, 7).astype("float32")   # mapa de baja resolución
    out = overlay_cam_on_image(rgb, cam, alpha=0.5)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    print("  [OK] overlay_cam_on_image (redimensiona y mezcla)")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Ejecutando pruebas de la Fase 3 (Grad-CAM):")
    test_target_layer()
    test_gradcam_with_frozen_backbone()
    test_overlay()
    print("TODAS LAS PRUEBAS DE FASE 3 PASARON.")
