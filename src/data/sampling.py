"""
Muestreo de frames a partir de vídeos.

El vídeo es caro de procesar entero, así que no usamos todos los frames: se
seleccionan N fotogramas equiespaciados por vídeo. Esta es la primera pieza de
la estrategia de "reducir el coste computacional" descrita en el alcance.

Solo depende de OpenCV y NumPy (no requiere PyTorch), por lo que puede usarse
en las primeras celdas del EDA sin cargar el stack de deep learning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def sample_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    """Devuelve los índices de N frames equiespaciados.

    Si el vídeo tiene menos frames que los pedidos, devuelve todos los disponibles.

    Args:
        total_frames: número total de frames del vídeo.
        num_frames: número de frames a muestrear.

    Returns:
        Lista de índices de frame (enteros, ordenados).
    """
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    return [int(i) for i in np.linspace(0, total_frames - 1, num_frames)]


def sample_frames(
    video_path: str | Path,
    num_frames: int = 16,
    as_rgb: bool = True,
) -> List[np.ndarray]:
    """Extrae N frames equiespaciados de un vídeo.

    Usa búsqueda por índice (seek) para no leer el vídeo entero. Los frames
    ilegibles o corruptos se omiten silenciosamente.

    Args:
        video_path: ruta al archivo de vídeo.
        num_frames: número de frames a muestrear.
        as_rgb: si True, convierte de BGR (OpenCV) a RGB.

    Returns:
        Lista de frames como arrays NumPy (alto, ancho, 3).

    Raises:
        IOError: si el vídeo no se puede abrir.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el vídeo: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total, num_frames)

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if as_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def get_video_metadata(video_path: str | Path) -> Dict[str, float]:
    """Lee metadatos básicos del vídeo (útil para el EDA).

    Args:
        video_path: ruta al archivo de vídeo.

    Returns:
        Diccionario con frame_count, fps, width, height y duration_sec.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "duration_sec": round(duration, 2),
    }
