"""
Extracción y cacheo de embeddings espaciales (Transfer Learning).

Una CNN preentrenada y CONGELADA (timm) convierte cada rostro recortado en un
vector de características. Esto es transfer learning en modo "feature extraction":
no se reentrena la CNN, solo se usa como extractor. Los embeddings se calculan
UNA SOLA VEZ y se cachean en data/processed/, de modo que el entrenamiento del
modelo temporal (Fase 2) opera sobre vectores ya calculados -> rápido y barato.

Salida por vídeo: un .npy de forma [n_rostros, embed_dim] en
    data/processed/<metodo>/<video_id>.npy
y un manifiesto CSV (data/processed/embeddings_manifest.csv) con las etiquetas,
métodos, splits y rutas, que consumirá el SequenceDataset.

Requiere PyTorch + timm.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def build_backbone(name: str = "efficientnet_b0", device: Optional[str] = None):
    """Crea la CNN preentrenada congelada y su transformación de entrada.

    Args:
        name: nombre del modelo en timm (p. ej. 'efficientnet_b0', 'resnet50').
        device: 'cuda' o 'cpu'. Si None, autodetecta.

    Returns:
        (model, transform, embed_dim, device). model devuelve el vector pooled
        (num_classes=0) y está en modo eval con los gradientes desactivados.
    """
    import timm
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # num_classes=0 -> el modelo devuelve directamente las características pooled.
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval().to(device)
    for p in model.parameters():  # congelar el backbone
        p.requires_grad_(False)

    # Transformación de entrada coherente con el preentrenamiento del modelo.
    data_cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_cfg)
    embed_dim = model.num_features
    return model, transform, embed_dim, device


def _frames_of_video(interim_method_dir: Path, video_id: str) -> List[Path]:
    """Devuelve, ordenadas, las rutas de los recortes faciales de un vídeo.

    Los archivos se nombran '<video_id>_frameNN.jpg'. Se usa '_frame' como
    separador para soportar ids con guion bajo (p. ej. '000_003').
    """
    return sorted(interim_method_dir.glob(f"{video_id}_frame*.jpg"))


def embed_video(
    model,
    transform,
    frame_paths: List[Path],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Calcula los embeddings de los frames de un vídeo.

    Returns:
        Array [n_frames, embed_dim] (vacío si no hay frames).
    """
    import torch

    if not frame_paths:
        return np.empty((0, model.num_features), dtype="float32")

    embeddings = []
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i : i + batch_size]
        imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model(batch)
        embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype("float32")


def build_embeddings(
    inventory: pd.DataFrame,
    interim_root: str | Path,
    processed_root: str | Path,
    backbone: str = "efficientnet_b0",
    batch_size: int = 32,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Calcula y cachea los embeddings de todo el inventario.

    Args:
        inventory: DataFrame de enumerate_videos (con columnas video_id, method,
            label y, opcionalmente, split).
        interim_root: carpeta data/interim (rostros recortados).
        processed_root: carpeta data/processed (destino de los .npy).
        backbone: modelo timm a usar.
        batch_size: tamaño de lote para el forward.
        overwrite: si False, omite vídeos ya cacheados.

    Returns:
        Manifiesto (DataFrame) con una fila por vídeo con embeddings.
    """
    interim_root = Path(interim_root)
    processed_root = Path(processed_root)

    model, transform, embed_dim, device = build_backbone(backbone)
    print(f"Backbone: {backbone} | embed_dim={embed_dim} | device={device}")

    records = []
    for row in tqdm(inventory.itertuples(index=False), total=len(inventory),
                    desc="Embeddings"):
        out_dir = processed_root / row.method
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{row.video_id}.npy"

        if out_path.exists() and not overwrite:
            emb = np.load(out_path)
        else:
            frames = _frames_of_video(interim_root / row.method, row.video_id)
            emb = embed_video(model, transform, frames, device, batch_size)
            if emb.shape[0] > 0:
                np.save(out_path, emb)

        if emb.shape[0] == 0:
            continue  # vídeos sin rostros detectados se descartan

        rec = {
            "video_id": row.video_id,
            "method": row.method,
            "label": int(row.label),
            "n_frames": int(emb.shape[0]),
            "embed_dim": int(emb.shape[1]),
            "embedding_path": str(out_path),
        }
        if hasattr(row, "split"):
            rec["split"] = getattr(row, "split")
        records.append(rec)

    manifest = pd.DataFrame(records)
    manifest_path = processed_root / "embeddings_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifiesto guardado: {manifest_path} ({len(manifest)} vídeos con embeddings).")
    return manifest
