"""
Utilidades de reproducibilidad y carga de configuración.

La guía del TFM valora explícitamente que el proyecto sea REPRODUCIBLE.
Fijar todas las semillas y centralizar la configuración es el primer paso.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fija todas las semillas relevantes para garantizar reproducibilidad.

    Cubre random, numpy y, si está disponible, PyTorch (CPU y GPU).

    Args:
        seed: valor de la semilla global.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Comportamiento determinista (puede ralentizar algo el entrenamiento):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch aún no instalado: no pasa nada, se fija el resto.
        pass


def load_config(path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    """Carga el fichero de configuración YAML como diccionario.

    Args:
        path: ruta al config.yaml.

    Returns:
        Diccionario con la configuración del proyecto.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """Devuelve 'cuda' si hay GPU disponible, en caso contrario 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
