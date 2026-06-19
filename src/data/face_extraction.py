"""
Detección y recorte facial sobre frames muestreados (MTCNN).

Para cada vídeo: se muestrean N frames y, en cada uno, se detecta el rostro
con MTCNN (facenet-pytorch) y se guarda el recorte normalizado en
data/interim/<metodo>/. Aislar el rostro reduce el ruido de fondo y enfoca al
modelo en la zona donde aparecen los artefactos del deepfake.

Este módulo SÍ requiere PyTorch + facenet-pytorch. La API usada:
    mtcnn(img, save_path=...) -> guarda el recorte si encuentra cara; None si no.

Uso por línea de comandos (procesa un subconjunto y para):
    python -m src.data.face_extraction --limit 20 --frames 16

IMPORTANTE: valida primero sobre 5-10 vídeos (con --limit) antes de lanzar el
dataset completo, tal y como se recomienda en el plan.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.dataset import enumerate_videos
from src.data.sampling import sample_frames
from src.utils.paths import get_paths
from src.utils.seeds import load_config, set_seed


def build_mtcnn(
    image_size: int = 224,
    margin: int = 20,
    device: Optional[str] = None,
    post_process: bool = False,
):
    """Crea el detector MTCNN.

    Args:
        image_size: tamaño del recorte de salida (px).
        margin: margen alrededor de la caja del rostro.
        device: 'cuda' o 'cpu'. Si None, se detecta automáticamente.
        post_process: si False, guarda el recorte sin normalizar (para visualizar
            y para cachear como imagen). La normalización se hará en el backbone.

    Returns:
        Instancia de MTCNN lista para usar.
    """
    import torch
    from facenet_pytorch import MTCNN

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # select_largest=True: si hay varias caras, se queda con la mayor (la principal).
    return MTCNN(
        image_size=image_size,
        margin=margin,
        post_process=post_process,
        select_largest=True,
        device=device,
    )


def extract_faces_from_video(
    video_path: str | Path,
    mtcnn,
    out_dir: str | Path,
    num_frames: int = 16,
    video_id: Optional[str] = None,
) -> int:
    """Muestrea frames de un vídeo, detecta y guarda los rostros.

    Args:
        video_path: ruta al vídeo.
        mtcnn: detector creado con build_mtcnn.
        out_dir: carpeta donde guardar los recortes.
        num_frames: nº de frames a muestrear.
        video_id: identificador para nombrar los archivos.

    Returns:
        Número de rostros efectivamente detectados y guardados.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = video_id or Path(video_path).stem

    frames = sample_frames(video_path, num_frames=num_frames, as_rgb=True)

    saved = 0
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        save_path = out_dir / f"{vid}_frame{i:02d}.jpg"
        # mtcnn devuelve el tensor del rostro (o None) y, con save_path, lo guarda.
        face = mtcnn(img, save_path=str(save_path))
        if face is not None:
            saved += 1
    return saved


def extract_dataset(
    df: pd.DataFrame,
    mtcnn,
    interim_root: str | Path,
    num_frames: int = 16,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Ejecuta la extracción facial sobre todo (o parte) del inventario.

    Args:
        df: inventario de enumerate_videos.
        mtcnn: detector.
        interim_root: carpeta data/interim.
        num_frames: nº de frames por vídeo.
        limit: si se indica, procesa solo los primeros N vídeos (para validar).

    Returns:
        DataFrame con el resultado por vídeo (n_faces detectadas).
    """
    interim_root = Path(interim_root)
    subset = df.head(limit) if limit else df

    records: List[dict] = []
    for row in tqdm(subset.itertuples(index=False), total=len(subset),
                    desc="Extrayendo rostros"):
        out_dir = interim_root / row.method
        n_faces = extract_faces_from_video(
            row.filepath, mtcnn, out_dir, num_frames=num_frames, video_id=row.video_id
        )
        records.append(
            {
                "video_id": row.video_id,
                "method": row.method,
                "label": row.label,
                "n_frames": num_frames,
                "n_faces": n_faces,
                "detection_rate": round(n_faces / num_frames, 3) if num_frames else 0.0,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracción facial sobre FF++.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N vídeos (validación).")
    parser.add_argument("--frames", type=int, default=None,
                        help="Frames por vídeo (sobrescribe el config).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    # Rutas resueltas: en Colab apuntan a Drive vía TFM_WORKSPACE; en local, al proyecto.
    paths = get_paths(cfg)
    root = paths["raw"]
    interim = paths["interim"]
    compression = cfg["dataset"]["compression"]
    methods = cfg["dataset"]["manipulation_methods"]
    num_frames = args.frames or cfg["face_extraction"]["frames_per_video"]

    df = enumerate_videos(root, compression=compression, methods=methods)
    if df.empty:
        print(f"[AVISO] No se encontraron vídeos en '{root}'. "
              f"¿Has descargado FF++ ({compression}) en esa ruta?")
        return

    print(f"Inventario: {len(df)} vídeos "
          f"({(df['label'] == 0).sum()} reales / {(df['label'] == 1).sum()} fakes).")

    mtcnn = build_mtcnn(
        image_size=cfg["face_extraction"]["image_size"],
        margin=cfg["face_extraction"]["margin"],
    )

    result = extract_dataset(df, mtcnn, interim, num_frames=num_frames, limit=args.limit)
    out_csv = Path(interim) / "extraction_report.csv"
    result.to_csv(out_csv, index=False)
    print(f"Hecho. Tasa media de detección: {result['detection_rate'].mean():.1%}. "
          f"Informe: {out_csv}")


if __name__ == "__main__":
    main()
