"""Pruebas del pipeline fusionado multi-backbone (vídeo -> embeddings).

Usa vídeos sintéticos y modelos simulados (sin descargas): valida la pasada
compartida de MTCNN, el almacenamiento por backbone, la reutilización del
manifiesto y la carga perezosa (segunda ejecución sin cargar modelos).
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import src.features.embeddings as E
import src.data.face_extraction as FX

DIMS = {"bbA": 8, "bbB": 12}


def make_video(path: Path, n=24, w=64, h=48):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, 25, (w, h))
    for i in range(n):
        wr.write(np.full((h, w, 3), i % 255, dtype=np.uint8))
    wr.release()


class FakeModel:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, batch):
        return torch.ones(batch.shape[0], self.dim)


def fake_build_backbone(name, device=None):
    return FakeModel(DIMS[name]), (lambda im: torch.zeros(3, 4, 4)), DIMS[name], "cpu"


def fake_build_mtcnn(image_size=224, margin=20, device=None, post_process=False):
    def mtcnn(x):
        if isinstance(x, list):                       # modo por lotes
            return [torch.rand(3, 8, 8) * 255 for _ in x]
        return torch.rand(3, 8, 8) * 255
    return mtcnn


def _make_inventory(tmp: Path) -> pd.DataFrame:
    rows = []
    for method, label, vids in [("original", 0, ["000", "001"]),
                                ("Deepfakes", 1, ["000_003", "001_004"])]:
        d = tmp / "videos" / method
        d.mkdir(parents=True)
        for vid in vids:
            vp = d / f"{vid}.mp4"
            make_video(vp)
            rows.append({"filepath": str(vp), "video_id": vid,
                         "method": method, "label": label, "split": "train"})
    return pd.DataFrame(rows)


def test_multi_backbone_and_lazy_skip():
    orig_bb, orig_mt = E.build_backbone, FX.build_mtcnn
    try:
        E.build_backbone = fake_build_backbone
        FX.build_mtcnn = fake_build_mtcnn

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            inv = _make_inventory(tmp)
            proc = tmp / "processed"

            # --- Primera pasada: 2 backbones a la vez (MTCNN compartido) ---
            mans = E.embed_videos_multi(inv, proc, backbone="bbA",
                                        extra_backbones=("bbB",),
                                        num_frames=6, image_size=8, margin=0)
            assert set(mans) == {"bbA", "bbB"}
            assert len(mans["bbA"]) == 4 and len(mans["bbB"]) == 4
            assert int(mans["bbA"]["embed_dim"].iloc[0]) == 8
            assert int(mans["bbB"]["embed_dim"].iloc[0]) == 12
            # Almacenamiento: principal en la raíz, extra en subcarpeta
            assert (proc / "original" / "000.npy").exists()
            assert (proc / "bbB" / "original" / "000.npy").exists()
            assert (proc / "embeddings_manifest.csv").exists()
            assert (proc / "bbB" / "embeddings_manifest.csv").exists()
            # video_id conserva ceros a la izquierda ('000', no 0)
            assert "000" in set(mans["bbA"]["video_id"])
            print("  [OK] multi-backbone: pasada compartida y almacenamiento correcto")

            # --- Segunda pasada: todo cacheado -> NO debe cargar modelos ---
            def boom(*a, **k):
                raise AssertionError("no debería cargar modelos con todo cacheado")
            E.build_backbone = boom
            FX.build_mtcnn = boom
            mans2 = E.embed_videos_multi(inv, proc, backbone="bbA",
                                         extra_backbones=("bbB",),
                                         num_frames=6, image_size=8, margin=0)
            assert len(mans2["bbA"]) == 4 and len(mans2["bbB"]) == 4
            assert "000" in set(mans2["bbA"]["video_id"])   # reuso del manifiesto (dtype str)
            print("  [OK] segunda pasada: todo cacheado, sin cargar modelos (perezoso)")

            # --- Backbone extra nuevo: solo procesa lo que falta de ese ---
            E.build_backbone = fake_build_backbone
            FX.build_mtcnn = fake_build_mtcnn
            DIMS["bbC"] = 5
            mans3 = E.embed_videos_multi(inv, proc, backbone="bbA",
                                         extra_backbones=("bbC",),
                                         num_frames=6, image_size=8, margin=0)
            assert len(mans3["bbC"]) == 4
            assert (proc / "bbC" / "Deepfakes" / "000_003.npy").exists()
            print("  [OK] backbone nuevo: calcula solo el pendiente, reutiliza el resto")
    finally:
        E.build_backbone = orig_bb
        FX.build_mtcnn = orig_mt


def test_scan_and_prev_manifest_helpers():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "original").mkdir()
        (root / "original" / "000.npy").touch()
        (root / "original" / "otro.txt").touch()
        done = E._scan_existing(root, ["original", "Deepfakes"])
        assert done == {("original", "000")}

        pd.DataFrame([{"video_id": "000", "method": "original", "label": 0,
                       "n_frames": 6, "embed_dim": 8,
                       "embedding_path": "x"}]).to_csv(
            root / "embeddings_manifest.csv", index=False)
        prev = E._load_prev_manifest(root)
        assert ("original", "000") in prev            # '000' no colapsa a 0
        assert int(prev[("original", "000")]["n_frames"]) == 6
        print("  [OK] _scan_existing + _load_prev_manifest (ids '000' preservados)")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Ejecutando pruebas del pipeline fusionado multi-backbone:")
    test_scan_and_prev_manifest_helpers()
    test_multi_backbone_and_lazy_skip()
    print("TODAS LAS PRUEBAS DEL PIPELINE FUSIONADO PASARON.")
