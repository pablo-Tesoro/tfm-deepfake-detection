"""
Dataset de secuencias de embeddings para el modelo temporal.

Cada vídeo es una secuencia de vectores [n_frames, embed_dim] (cacheada en
data/processed). Este Dataset las sirve con longitud fija (pad/truncado) y
conserva la longitud real para poder "empaquetar" la secuencia en la RNN.

Incluye constructores que respetan:
  - el split oficial train/val/test (comparabilidad con la literatura);
  - el experimento CROSS-MANIPULATION (entrenar con 3 métodos y evaluar en el 4º
    no visto), gran diferenciador del TFM.

Requiere PyTorch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _import_torch():
    import torch
    from torch.utils.data import Dataset
    return torch, Dataset


# Definición perezosa para no exigir torch al importar el módulo.
torch, _Dataset = _import_torch()


class SequenceDataset(_Dataset):
    """Sirve (secuencia_embeddings, longitud, etiqueta) por vídeo."""

    def __init__(self, manifest: pd.DataFrame, max_len: int = 16):
        # Solo vídeos con al menos un rostro detectado.
        self.df = manifest[manifest["n_frames"] > 0].reset_index(drop=True)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        emb = np.load(row["embedding_path"]).astype("float32")  # [n, D]
        n = emb.shape[0]

        if n >= self.max_len:
            emb = emb[: self.max_len]
            length = self.max_len
        else:
            pad = np.zeros((self.max_len - n, emb.shape[1]), dtype="float32")
            emb = np.concatenate([emb, pad], axis=0)
            length = n

        return (
            torch.from_numpy(emb),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(int(row["label"]), dtype=torch.float32),
        )


def collate_sequences(batch):
    """Agrupa un lote en tensores: X [B,T,D], lengths [B], labels [B]."""
    xs, lengths, labels = zip(*batch)
    return torch.stack(xs), torch.stack(lengths), torch.stack(labels)


def load_manifest(processed_root: str | Path) -> pd.DataFrame:
    """Carga el manifiesto de embeddings."""
    return pd.read_csv(Path(processed_root) / "embeddings_manifest.csv")


def split_by_official(
    manifest: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Divide el manifiesto según la columna 'split' (train/val/test)."""
    if "split" not in manifest.columns:
        raise ValueError("El manifiesto no tiene columna 'split'. "
                         "Asigna los splits oficiales antes de modelar.")
    return {
        name: manifest[manifest["split"] == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }


def split_cross_manipulation(
    manifest: pd.DataFrame,
    train_methods: Sequence[str],
    holdout_method: str,
) -> Dict[str, pd.DataFrame]:
    """Prepara el experimento de generalización a un método NO visto.

    Entrenamiento/validación: vídeos reales + fakes de `train_methods`.
    Test: vídeos reales + fakes del `holdout_method` (no visto en entrenamiento).
    Se respeta el split oficial para repartir los vídeos reales y, en train,
    se usa el split 'val' para validación.

    Returns:
        Diccionario con 'train', 'val', 'test'.
    """
    reals = manifest[manifest["label"] == 0]
    fakes = manifest[manifest["label"] == 1]

    train_fakes = fakes[fakes["method"].isin(train_methods)]
    holdout_fakes = fakes[fakes["method"] == holdout_method]

    has_split = "split" in manifest.columns

    if has_split:
        train = pd.concat([
            reals[reals["split"] == "train"],
            train_fakes[train_fakes["split"] == "train"],
        ])
        val = pd.concat([
            reals[reals["split"] == "val"],
            train_fakes[train_fakes["split"] == "val"],
        ])
        test = pd.concat([
            reals[reals["split"] == "test"],
            holdout_fakes[holdout_fakes["split"] == "test"],
        ])
    else:
        # Sin split oficial: reparto simple (no recomendado para la entrega).
        train = pd.concat([reals, train_fakes])
        val = train.sample(frac=0.15, random_state=42)
        train = train.drop(val.index)
        test = pd.concat([reals, holdout_fakes])

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def class_weights(manifest: pd.DataFrame) -> float:
    """Calcula el peso de la clase positiva (fake) para BCEWithLogitsLoss.

    pos_weight = n_negativos / n_positivos. Útil para mitigar el desbalance.
    """
    n_pos = int((manifest["label"] == 1).sum())
    n_neg = int((manifest["label"] == 0).sum())
    return (n_neg / n_pos) if n_pos > 0 else 1.0
