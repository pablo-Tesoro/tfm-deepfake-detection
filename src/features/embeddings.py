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

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _scan_existing(root: Path, methods: Sequence[str]) -> set:
    """(method, video_id) con .npy ya cacheado.

    UNA lectura de directorio por método: crítico en Google Drive, donde miles
    de comprobaciones de existencia individuales tardan horas.
    """
    done = set()
    for m in methods:
        d = root / m
        if not d.exists():
            continue
        try:
            names = os.listdir(d)
        except OSError:
            names = []
        for n in names:
            if n.endswith(".npy"):
                done.add((m, n[:-4]))
    return done


def _load_prev_manifest(root: Path) -> dict:
    """Manifiesto previo como {(method, video_id): fila}.

    Permite reutilizar metadatos (n_frames, embed_dim) de vídeos ya procesados
    sin releer miles de .npy pequeños de Drive en cada re-ejecución.
    """
    p = root / "embeddings_manifest.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, dtype={"video_id": str})
        return {(r["method"], str(r["video_id"])): r for _, r in df.iterrows()}
    except Exception:
        return {}


def _detect_faces(mtcnn, frames) -> List:
    """Detecta el rostro de cada frame y devuelve los recortes como PIL.

    Usa el modo por LOTES de MTCNN (mucho más rápido en GPU: una pasada para
    los 16 frames en vez de 16) y cae a frame a frame si no está soportado.
    """
    pil = [Image.fromarray(f) for f in frames]
    if not pil:
        return []
    try:
        outs = mtcnn(pil)
    except Exception:
        outs = [mtcnn(p) for p in pil]
    if not isinstance(outs, list):
        outs = [outs]
    faces = []
    for t in outs:
        if t is None:
            continue
        img = t.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        faces.append(Image.fromarray(img))
    return faces


def embed_videos_multi(
    inventory: pd.DataFrame,
    processed_root: str | Path,
    backbone: str = "efficientnet_b0",
    extra_backbones: Sequence[str] = (),
    num_frames: int = 16,
    image_size: int = 224,
    margin: int = 20,
    overwrite: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Vídeo -> embedding para 1..N backbones EN UNA SOLA PASADA.

    La detección facial (MTCNN, la parte cara) se hace UNA vez por vídeo y sus
    rostros alimentan a todos los backbones pendientes. Optimizaciones clave
    para Google Drive:
      - no escribe frames: solo un .npy por vídeo y backbone;
      - detección facial por lotes;
      - escaneo de cada directorio una única vez + reutilización del manifiesto
        previo (sin releer .npy ya hechos);
      - carga perezosa: si todo está cacheado, no carga modelos ni toca la GPU.

    Almacenamiento: el backbone principal en processed_root/<metodo>/;
    los adicionales en processed_root/<backbone>/<metodo>/.

    Returns:
        {backbone: manifiesto} (cada manifiesto se guarda también como CSV).
    """
    import torch

    processed_root = Path(processed_root)
    backbones = [backbone] + [b for b in extra_backbones if b and b != backbone]
    roots = {bb: (processed_root if i == 0 else processed_root / bb)
             for i, bb in enumerate(backbones)}
    methods = list(dict.fromkeys(inventory["method"]))

    done = {bb: (set() if overwrite else _scan_existing(roots[bb], methods))
            for bb in backbones}
    prev = {bb: _load_prev_manifest(roots[bb]) for bb in backbones}

    pending = sum(
        1 for row in inventory.itertuples(index=False)
        if any((row.method, str(row.video_id)) not in done[bb] for bb in backbones)
    )

    models: Dict[str, tuple] = {}
    mtcnn = None
    device = "cpu"
    if pending:
        from src.data.face_extraction import build_mtcnn
        for bb in backbones:
            model, transform, dim, device = build_backbone(bb)
            models[bb] = (model, transform)
        mtcnn = build_mtcnn(image_size=image_size, margin=margin, device=device)
        print(f"Fusionado | backbones={backbones} | device={device} | "
              f"pendientes={pending}/{len(inventory)}")
    else:
        print(f"Fusionado | backbones={backbones} | todo cacheado "
              f"({len(inventory)} vídeos): no se cargan modelos.")

    from src.data.sampling import sample_frames

    records: Dict[str, list] = {bb: [] for bb in backbones}
    iterator = inventory.itertuples(index=False)
    if pending:
        iterator = tqdm(iterator, total=len(inventory), desc="Video -> embedding")

    for row in iterator:
        vid = str(row.video_id)
        missing = [bb for bb in backbones if (row.method, vid) not in done[bb]]

        faces = None
        if missing:
            try:
                frames = sample_frames(row.filepath, num_frames=num_frames, as_rgb=True)
            except Exception:
                frames = []
            faces = _detect_faces(mtcnn, frames)

        for bb in backbones:
            out_path = roots[bb] / row.method / f"{vid}.npy"
            if bb in missing:
                if not faces:
                    continue                              # sin rostro: se descarta
                model, transform = models[bb]
                batch = torch.stack([transform(f) for f in faces]).to(device)
                with torch.no_grad():
                    emb = model(batch).cpu().numpy().astype("float32")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, emb)
                n_f, e_d = int(emb.shape[0]), int(emb.shape[1])
            else:
                prev_row = prev[bb].get((row.method, vid))
                if prev_row is not None:                  # metadatos sin releer .npy
                    n_f, e_d = int(prev_row["n_frames"]), int(prev_row["embed_dim"])
                else:                                     # fallback poco frecuente
                    try:
                        shape = np.load(out_path, mmap_mode="r").shape
                        n_f, e_d = int(shape[0]), int(shape[1])
                    except Exception:
                        continue
            rec = {"video_id": vid, "method": row.method, "label": int(row.label),
                   "n_frames": n_f, "embed_dim": e_d,
                   "embedding_path": str(out_path)}
            if hasattr(row, "split"):
                rec["split"] = getattr(row, "split")
            records[bb].append(rec)

    manifests: Dict[str, pd.DataFrame] = {}
    for bb in backbones:
        man = pd.DataFrame(records[bb])
        roots[bb].mkdir(parents=True, exist_ok=True)
        man.to_csv(roots[bb] / "embeddings_manifest.csv", index=False)
        manifests[bb] = man
        print(f"  {bb}: {len(man)} vídeos con embeddings")
    return manifests


def embed_videos_direct(
    inventory: pd.DataFrame,
    processed_root: str | Path,
    backbone: str = "efficientnet_b0",
    num_frames: int = 16,
    image_size: int = 224,
    margin: int = 20,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Atajo de un solo backbone sobre embed_videos_multi (mismas optimizaciones)."""
    return embed_videos_multi(
        inventory, processed_root, backbone=backbone, extra_backbones=(),
        num_frames=num_frames, image_size=image_size, margin=margin,
        overwrite=overwrite)[backbone]


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
