"""Pruebas con datos sintéticos de la lógica que no requiere PyTorch."""
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sampling import sample_frame_indices, sample_frames, get_video_metadata
from src.data.dataset import enumerate_videos, load_official_splits, assign_splits
from src.utils.paths import get_paths, resolve_workspace


def make_synthetic_video(path: Path, n_frames: int, w: int = 64, h: int = 48, fps: int = 25):
    """Crea un vídeo sintético (frames de colores) para probar el muestreo."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), (i % 256, (2 * i) % 256, (3 * i) % 256), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_sample_frame_indices():
    assert sample_frame_indices(0, 16) == []
    assert sample_frame_indices(5, 16) == [0, 1, 2, 3, 4]          # menos frames que pedidos
    idx = sample_frame_indices(100, 16)
    assert len(idx) == 16
    assert idx[0] == 0 and idx[-1] == 99                            # incluye extremos
    assert idx == sorted(idx)                                       # ordenados
    print("  [OK] sample_frame_indices")


def test_sampling_and_metadata():
    with tempfile.TemporaryDirectory() as d:
        vp = Path(d) / "fake.mp4"
        make_synthetic_video(vp, n_frames=120, w=80, h=60, fps=30)

        frames = sample_frames(vp, num_frames=16)
        assert len(frames) == 16, f"esperaba 16 frames, obtuve {len(frames)}"
        assert frames[0].shape == (60, 80, 3)                       # alto, ancho, canales

        meta = get_video_metadata(vp)
        assert meta["width"] == 80 and meta["height"] == 60
        assert meta["frame_count"] == 120
        assert abs(meta["fps"] - 30) < 1
        print("  [OK] sample_frames + get_video_metadata")


def test_path_resolution():
    cfg = {"paths": {"workspace_root": None}}

    # 1) Por defecto (sin env, sin config): workspace = project_root
    os.environ.pop("TFM_WORKSPACE", None)
    paths = get_paths(cfg, "/proj")
    assert str(paths["raw"]) == "/proj/data/raw"
    assert str(paths["figures"]) == "/proj/reports/figures"
    assert str(paths["models"]) == "/proj/models"

    # 2) Override por variable de entorno (caso Colab/Drive)
    os.environ["TFM_WORKSPACE"] = "/content/drive/MyDrive/TFM_Deepfake"
    paths = get_paths(cfg, "/proj")
    assert str(paths["raw"]) == "/content/drive/MyDrive/TFM_Deepfake/data/raw"
    assert str(paths["processed"]).startswith("/content/drive/MyDrive/TFM_Deepfake")
    os.environ.pop("TFM_WORKSPACE", None)

    # 3) Override por config
    cfg2 = {"paths": {"workspace_root": "/mnt/ws"}}
    paths = get_paths(cfg2, "/proj")
    assert str(paths["interim"]) == "/mnt/ws/data/interim"
    print("  [OK] get_paths (local / Colab-Drive / config)")


def test_dataset_enumeration_and_splits():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Estructura FF++ sintética
        orig = root / "original_sequences" / "youtube" / "c23" / "videos"
        orig.mkdir(parents=True)
        for vid in ("000", "001", "002"):
            (orig / f"{vid}.mp4").touch()

        for method in ("Deepfakes", "Face2Face"):
            man = root / "manipulated_sequences" / method / "c23" / "videos"
            man.mkdir(parents=True)
            for vid in ("000_003", "001_004"):
                (man / f"{vid}.mp4").touch()

        df = enumerate_videos(root, compression="c23",
                              methods=("Deepfakes", "Face2Face"))
        # 3 reales + 2 métodos * 2 fakes = 7
        assert len(df) == 7, f"esperaba 7 vídeos, obtuve {len(df)}"
        assert (df["label"] == 0).sum() == 3                        # reales
        assert (df["label"] == 1).sum() == 4                        # fakes
        assert set(df["source_id"]) == {"000", "001", "002"}        # id de origen
        assert df[df["category"] == "manipulated"]["method"].nunique() == 2

        # Splits sintéticos
        splits_dir = root / "splits"
        splits_dir.mkdir()
        (splits_dir / "train.json").write_text(json.dumps([["000", "003"]]))
        (splits_dir / "val.json").write_text(json.dumps([["001", "004"]]))
        (splits_dir / "test.json").write_text(json.dumps([["002", "005"]]))

        splits = load_official_splits(splits_dir)
        df["split"] = assign_splits(df, splits)
        # source_id 000 -> train, 001 -> val, 002 -> test
        assert df.loc[df["source_id"] == "000", "split"].eq("train").all()
        assert df.loc[df["source_id"] == "001", "split"].eq("val").all()
        assert df.loc[df["source_id"] == "002", "split"].eq("test").all()
        print("  [OK] enumerate_videos + splits")


if __name__ == "__main__":
    print("Ejecutando pruebas de la Fase 1 (sin PyTorch):")
    test_sample_frame_indices()
    test_sampling_and_metadata()
    test_path_resolution()
    test_dataset_enumeration_and_splits()
    print("TODAS LAS PRUEBAS PASARON.")
