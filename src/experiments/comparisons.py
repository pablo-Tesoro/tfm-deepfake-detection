"""
Experimentos avanzados para enriquecer la sección de modelización del TFM.

Tres análisis, todos reutilizando el pipeline ya probado:

1. learning_curve  -> progresión del AUC según el nº de vídeos de entrenamiento.
2. compare_backbones -> EfficientNet-B0 vs ResNet-50 (mismas condiciones).
3. per_method_metrics -> rendimiento desglosado por método de manipulación.

El conjunto de TEST se mantiene fijo en todos los casos para que las comparaciones
sean justas. Requiere PyTorch + scikit-learn (+ timm para compare_backbones).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _build_loader(df: pd.DataFrame, max_len: int, batch: int, shuffle: bool):
    from torch.utils.data import DataLoader
    from src.data.sequence_dataset import SequenceDataset, collate_sequences
    return DataLoader(SequenceDataset(df, max_len), batch_size=batch,
                      shuffle=shuffle, collate_fn=collate_sequences)


def train_eval_hybrid(parts: Dict[str, pd.DataFrame], embed_dim: int, cfg: dict,
                      device: str, epochs: Optional[int] = None, verbose: bool = False):
    """Entrena un híbrido CNN+LSTM y lo evalúa en test. Devuelve (modelo, y, prob)."""
    from src.models.hybrid import CNNLSTM
    from src.training.trainer import train_model, _predict_proba
    from src.data.sequence_dataset import class_weights

    max_len = cfg["face_extraction"]["frames_per_video"]
    batch = cfg["training"]["batch_size"]
    tr = _build_loader(parts["train"], max_len, batch, True)
    va = _build_loader(parts["val"], max_len, batch, False)
    te = _build_loader(parts["test"], max_len, batch, False)

    model = CNNLSTM(embed_dim, hidden=cfg["model"]["hidden_size"],
                    num_layers=cfg["model"]["num_layers"],
                    rnn_type=cfg["model"]["temporal"], dropout=cfg["model"]["dropout"])
    train_model(model, tr, va, epochs=epochs or cfg["training"]["epochs"],
                lr=cfg["training"]["learning_rate"], pos_weight=class_weights(parts["train"]),
                device=device, patience=5, verbose=verbose)
    y, prob = _predict_proba(model, te, device)
    return model, y, prob


def subsample_stratified(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Submuestra n vídeos manteniendo la proporción de clases."""
    from sklearn.model_selection import train_test_split
    if n >= len(df):
        return df
    try:
        keep, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    except ValueError:  # alguna clase demasiado pequeña para estratificar
        keep = df.sample(n=n, random_state=seed)
    return keep.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1) Curva de aprendizaje: AUC vs nº de vídeos
# ---------------------------------------------------------------------------
def learning_curve(manifest: pd.DataFrame, cfg: dict, device: str,
                   sizes: Optional[Sequence[int]] = None, seed: int = 42) -> pd.DataFrame:
    """Entrena con subconjuntos de entrenamiento crecientes y mide el AUC en test.

    El test y la validación se mantienen FIJOS; solo crece el entrenamiento.

    Returns:
        DataFrame con columnas n_videos, auc, f1, accuracy.
    """
    from src.data.sequence_dataset import get_splits
    from src.evaluation.metrics import compute_metrics

    parts = get_splits(manifest)
    embed_dim = int(manifest["embed_dim"].iloc[0])
    train_pool, val_df, test_df = parts["train"], parts["val"], parts["test"]

    if sizes is None:
        total = len(train_pool)
        sizes = sorted({max(10, int(total * f)) for f in (0.2, 0.4, 0.6, 0.8, 1.0)})

    rows = []
    for n in sizes:
        sub_train = subsample_stratified(train_pool, n, seed)
        sub_parts = {"train": sub_train, "val": val_df, "test": test_df}
        _, y, prob = train_eval_hybrid(sub_parts, embed_dim, cfg, device)
        m = compute_metrics(y, prob)
        rows.append({"n_videos": len(sub_train), "auc": m["auc"],
                     "f1": m["f1"], "accuracy": m["accuracy"]})
        print(f"  n_train={len(sub_train):4d} -> AUC={m['auc']:.3f} | F1={m['f1']:.3f}")
    return pd.DataFrame(rows)


def plot_learning_curve(df: pd.DataFrame, save_path: Optional[str | Path] = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df["n_videos"], df["auc"], marker="o", linewidth=2.2,
            color="#2a9d8f", label="AUC")
    ax.plot(df["n_videos"], df["f1"], marker="s", linewidth=1.8,
            color="#e76f51", alpha=.8, label="F1")
    ax.axhline(0.5, ls="--", color="gray", alpha=.6)
    ax.set_xlabel("nº de vídeos de entrenamiento")
    ax.set_ylabel("rendimiento en test")
    ax.set_title("Progresión del rendimiento según el tamaño del dataset")
    ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2) Comparación de backbones: EfficientNet vs ResNet
# ---------------------------------------------------------------------------
def get_backbone_manifest(inventory: pd.DataFrame, processed_root: str | Path,
                          backbone: str, cfg: dict) -> pd.DataFrame:
    """Embeddings de un backbone concreto (pipeline fusionado vídeo -> .npy).

    El backbone principal (el del config) vive en processed/; los alternativos
    en processed/<backbone>/. Idempotente: si ya está todo cacheado, no carga
    modelos ni toca la GPU.
    """
    from src.features.embeddings import embed_videos_multi

    fe = cfg["face_extraction"]
    root = Path(processed_root)
    if backbone != cfg["model"]["backbone"]:
        root = root / backbone
    return embed_videos_multi(
        inventory, root, backbone=backbone,
        num_frames=fe["frames_per_video"], image_size=fe["image_size"],
        margin=fe["margin"])[backbone]


def compare_backbones(inventory: pd.DataFrame, processed_root: str | Path,
                      cfg: dict, device: str,
                      backbones: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Entrena y evalúa el híbrido con cada backbone, en igualdad de condiciones.

    Para cada backbone: embeddings fusionados (cacheados) -> mismo split ->
    híbrido entrenado idéntico -> métricas en test. OJO: un backbone nuevo
    implica una pasada vídeo->embedding (MTCNN + CNN) la primera vez; después
    queda cacheado y la comparación es casi instantánea.

    Returns:
        DataFrame indexado por backbone con accuracy, precision, recall, f1, auc.
    """
    from src.data.sequence_dataset import get_splits
    from src.evaluation.metrics import compute_metrics

    backbones = backbones or [cfg["model"]["backbone"], "resnet50"]
    rows = []
    for bb in backbones:
        print(f"\n=== Backbone: {bb} ===")
        manifest = get_backbone_manifest(inventory, processed_root, bb, cfg)
        if manifest.empty:
            print("  (sin embeddings; se omite)")
            continue
        parts = get_splits(manifest)
        embed_dim = int(manifest["embed_dim"].iloc[0])
        _, y, prob = train_eval_hybrid(parts, embed_dim, cfg, device)
        m = compute_metrics(y, prob)
        m["backbone"] = bb
        m["embed_dim"] = embed_dim
        rows.append(m)
        print(f"  -> AUC={m['auc']:.3f} | F1={m['f1']:.3f} | dim={embed_dim}")

    df = pd.DataFrame(rows).set_index("backbone")
    cols = ["embed_dim", "accuracy", "precision", "recall", "f1", "auc"]
    return df[cols].round(4)


# ---------------------------------------------------------------------------
# 3) Métricas por método de manipulación
# ---------------------------------------------------------------------------
def per_method_metrics(manifest: pd.DataFrame, model, cfg: dict, device: str) -> pd.DataFrame:
    """Evalúa un modelo entrenado por separado en cada método (sobre el test).

    Para cada método M: vídeos reales de test + fakes de M en test. Revela qué
    manipulaciones son más fáciles o difíciles de detectar.

    Returns:
        DataFrame indexado por método con accuracy, precision, recall, f1, auc.
    """
    from src.data.sequence_dataset import get_splits
    from src.training.trainer import _predict_proba
    from src.evaluation.metrics import compute_metrics, metrics_table

    parts = get_splits(manifest)
    test = parts["test"]
    max_len = cfg["face_extraction"]["frames_per_video"]
    batch = cfg["training"]["batch_size"]

    reals = test[test["label"] == 0]
    results = {}
    for method in cfg["dataset"]["manipulation_methods"]:
        fakes = test[(test["label"] == 1) & (test["method"] == method)]
        if fakes.empty or reals.empty:
            continue
        subset = pd.concat([reals, fakes])
        loader = _build_loader(subset, max_len, batch, shuffle=False)
        y, prob = _predict_proba(model, loader, device)
        results[method] = compute_metrics(y, prob)

    return metrics_table(results)
