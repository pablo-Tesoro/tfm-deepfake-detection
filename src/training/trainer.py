"""
Bucle de entrenamiento genérico para los modelos de detección.

Sirve tanto para el baseline como para el híbrido (misma interfaz forward:
modelo(x, lengths) -> logits). Incluye:
  - BCEWithLogitsLoss con pos_weight opcional (para el desbalance de clases);
  - early stopping según el AUC de validación;
  - guardado del mejor checkpoint en models/.

Requiere PyTorch + scikit-learn.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (y_true, y_prob) recorriendo un DataLoader."""
    model.eval()
    ys, ps = [], []
    for x, lengths, y in loader:
        x, lengths = x.to(device), lengths.to(device)
        logits = model(x, lengths)
        probs = torch.sigmoid(logits).cpu().numpy()
        ps.append(probs)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-4,
    pos_weight: Optional[float] = None,
    device: Optional[str] = None,
    patience: int = 5,
    checkpoint_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Dict:
    """Entrena un modelo y devuelve el historial y el mejor estado.

    Args:
        model: modelo con forward(x, lengths) -> logits.
        train_loader, val_loader: DataLoaders de secuencias.
        epochs: épocas máximas.
        lr: learning rate (Adam).
        pos_weight: peso de la clase positiva (desbalance). None = sin peso.
        device: 'cuda'/'cpu'. None autodetecta.
        patience: épocas sin mejorar el AUC de val antes de parar.
        checkpoint_path: si se indica, guarda ahí el mejor modelo.
        verbose: imprime el progreso.

    Returns:
        Dict con 'history' (listas por época) y 'best_val_auc'.
    """
    from sklearn.metrics import roc_auc_score

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pw = torch.tensor([pos_weight], device=device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        # Validación
        y_true, y_prob = _predict_proba(model, val_loader, device)
        val_loss = nn.functional.binary_cross_entropy(
            torch.tensor(y_prob).clamp(1e-7, 1 - 1e-7), torch.tensor(y_true)
        ).item()
        # AUC requiere ambas clases presentes en validación
        val_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if verbose:
            print(f"Época {epoch:02d} | train_loss={train_loss:.4f} "
                  f"| val_loss={val_loss:.4f} | val_AUC={val_auc:.4f}")

        # Early stopping por AUC de validación
        if np.isnan(val_auc):
            current = -val_loss  # si no hay AUC, usar -val_loss como criterio
        else:
            current = val_auc
        if current > best_auc:
            best_auc = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping en la época {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
            if verbose:
                print(f"Mejor modelo guardado en {checkpoint_path}")

    return {"history": history, "best_val_auc": best_auc}
