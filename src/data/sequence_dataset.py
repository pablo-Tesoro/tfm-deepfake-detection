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


def split_stratified(
    manifest: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Reparto ESTRATIFICADO por clase sobre los vídeos disponibles.

    Garantiza que train/val/test contengan ambas clases. Es la opción adecuada
    para subconjuntos pequeños, donde el split oficial puede dejar conjuntos
    vacíos (p. ej. ningún vídeo de test).
    """
    from sklearn.model_selection import train_test_split

    df = manifest[manifest["n_frames"] > 0].reset_index(drop=True)
    train_df, tmp_df = train_test_split(
        df, test_size=val_frac + test_frac, stratify=df["label"], random_state=seed
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        tmp_df, test_size=rel_test, stratify=tmp_df["label"], random_state=seed
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def splits_usable(parts: Dict[str, pd.DataFrame], min_per_split: int = 2) -> bool:
    """True si los tres conjuntos tienen suficientes vídeos y ambas clases."""
    for name in ("train", "val", "test"):
        part = parts.get(name)
        if part is None or len(part) < min_per_split or part["label"].nunique() < 2:
            return False
    return True


def get_splits(
    manifest: pd.DataFrame,
    prefer_official: bool = True,
    min_per_split: int = 2,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Devuelve el split oficial si es utilizable; si no, uno estratificado.

    Pensado para que el notebook funcione tanto con el dataset completo (split
    oficial, comparable con la literatura) como con subconjuntos pequeños.
    """
    if prefer_official and "split" in manifest.columns:
        official = split_by_official(manifest)
        if splits_usable(official, min_per_split):
            print("Reparto: split OFICIAL de FaceForensics++.")
            return official
        print("Aviso: el split oficial deja conjuntos vacíos o insuficientes en "
              "este subconjunto -> se usa un reparto ESTRATIFICADO.")
    return split_stratified(manifest, **kwargs)


def split_cross_manipulation(
    manifest: pd.DataFrame,
    train_methods: Sequence[str],
    holdout_method: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Prepara el experimento de generalización a un método NO visto.

    Invariante clave: el `holdout_method` NO aparece en train ni en val; solo en
    test. Los vídeos reales se reparten de forma estratificada entre los tres
    conjuntos, de modo que el experimento funciona también con subconjuntos
    pequeños (sin depender de que el split oficial cubra cada conjunto).

    Returns:
        Diccionario con 'train', 'val', 'test'.
    """
    from sklearn.model_selection import train_test_split

    m = manifest[manifest["n_frames"] > 0]
    reals = m[m["label"] == 0]
    train_fakes = m[(m["label"] == 1) & (m["method"].isin(train_methods))]
    holdout_fakes = m[(m["label"] == 1) & (m["method"] == holdout_method)]

    # Reales -> train/val/test
    r_train, r_tmp = train_test_split(reals, test_size=val_frac + test_frac, random_state=seed)
    rel_test = test_frac / (val_frac + test_frac)
    r_val, r_test = train_test_split(r_tmp, test_size=rel_test, random_state=seed)

    # Fakes de métodos de entrenamiento -> train/val (test usa SOLO el holdout)
    f_train, f_val = train_test_split(train_fakes, test_size=val_frac, random_state=seed)

    return {
        "train": pd.concat([r_train, f_train]).reset_index(drop=True),
        "val": pd.concat([r_val, f_val]).reset_index(drop=True),
        "test": pd.concat([r_test, holdout_fakes]).reset_index(drop=True),
    }


def class_weights(manifest: pd.DataFrame) -> float:
    """Calcula el peso de la clase positiva (fake) para BCEWithLogitsLoss.

    pos_weight = n_negativos / n_positivos. Útil para mitigar el desbalance.
    """
    n_pos = int((manifest["label"] == 1).sum())
    n_neg = int((manifest["label"] == 0).sum())
    return (n_neg / n_pos) if n_pos > 0 else 1.0
