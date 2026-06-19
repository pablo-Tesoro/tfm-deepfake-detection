"""
Navegación del dataset FaceForensics++.

FF++ tiene una estructura de carpetas concreta. Este módulo la recorre y
construye un inventario (DataFrame) con la ruta de cada vídeo, su etiqueta
(real=0 / fake=1), el método de manipulación y el split oficial.

Estructura esperada tras la descarga (compresión c23):

    <root>/
    ├── original_sequences/youtube/c23/videos/000.mp4 ...        (REALES)
    └── manipulated_sequences/
        ├── Deepfakes/c23/videos/000_003.mp4 ...                 (FAKES)
        ├── Face2Face/c23/videos/...
        ├── FaceSwap/c23/videos/...
        └── NeuralTextures/c23/videos/...

Los nombres de los manipulados son "target_source" (p. ej. 000_003), por lo que
el id de origen para el split es la primera parte ("000").

Solo depende de pandas (no requiere PyTorch).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Set

import pandas as pd

REAL_LABEL = 0
FAKE_LABEL = 1

DEFAULT_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def enumerate_videos(
    root: str | Path,
    compression: str = "c23",
    methods: Sequence[str] = DEFAULT_METHODS,
) -> pd.DataFrame:
    """Construye el inventario de vídeos del dataset.

    Args:
        root: carpeta raíz de FaceForensics++.
        compression: nivel de compresión ('raw', 'c23', 'c40').
        methods: métodos de manipulación a incluir.

    Returns:
        DataFrame con columnas: filepath, label, category, method, video_id,
        source_id.
    """
    root = Path(root)
    rows = []

    # --- Vídeos reales (original_sequences) ---
    orig_dir = root / "original_sequences" / "youtube" / compression / "videos"
    for vp in sorted(orig_dir.glob("*.mp4")):
        rows.append(
            {
                "filepath": str(vp),
                "label": REAL_LABEL,
                "category": "original",
                "method": "original",
                "video_id": vp.stem,
            }
        )

    # --- Vídeos manipulados (manipulated_sequences) ---
    for method in methods:
        man_dir = root / "manipulated_sequences" / method / compression / "videos"
        for vp in sorted(man_dir.glob("*.mp4")):
            rows.append(
                {
                    "filepath": str(vp),
                    "label": FAKE_LABEL,
                    "category": "manipulated",
                    "method": method,
                    "video_id": vp.stem,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        # id de origen para casar con el split oficial (primera parte del nombre)
        df["source_id"] = df["video_id"].str.split("_").str[0]
    return df


def load_official_splits(splits_dir: str | Path) -> Dict[str, Set[str]]:
    """Carga los splits oficiales (train/val/test) de FF++.

    Los ficheros train.json, val.json y test.json contienen listas de pares
    de ids de vídeo. Se incluye 'dataset/splits/' en el repo oficial de FF++.

    Args:
        splits_dir: carpeta que contiene train.json, val.json, test.json.

    Returns:
        Diccionario {split: conjunto de ids de vídeo}.
    """
    splits_dir = Path(splits_dir)
    result: Dict[str, Set[str]] = {}
    for name in ("train", "val", "test"):
        with open(splits_dir / f"{name}.json", "r", encoding="utf-8") as f:
            pairs = json.load(f)
        ids: Set[str] = set()
        for pair in pairs:
            ids.update(str(p) for p in pair)
        result[name] = ids
    return result


def assign_splits(df: pd.DataFrame, splits: Dict[str, Set[str]]) -> pd.Series:
    """Asigna a cada vídeo su split oficial a partir de su source_id.

    Args:
        df: inventario devuelto por enumerate_videos.
        splits: diccionario de load_official_splits.

    Returns:
        Serie con el split ('train'/'val'/'test'/NaN) de cada fila.
    """
    id_to_split: Dict[str, str] = {}
    for name, ids in splits.items():
        for i in ids:
            id_to_split[i] = name
    return df["source_id"].map(id_to_split)
