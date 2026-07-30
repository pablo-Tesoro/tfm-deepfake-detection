"""Pruebas de los experimentos (curva de aprendizaje, por método). Requiere torch + sklearn."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.experiments.comparisons import (
    subsample_stratified, train_eval_hybrid, learning_curve, per_method_metrics,
)

EMBED_DIM = 32
MAX_LEN = 16
CFG = {
    "face_extraction": {"frames_per_video": MAX_LEN},
    "training": {"batch_size": 16, "epochs": 5, "learning_rate": 1e-3},
    "model": {"hidden_size": 16, "num_layers": 1, "temporal": "lstm", "dropout": 0.3},
    "dataset": {"manipulation_methods": ["Deepfakes", "Face2Face", "NeuralTextures"]},
}


def _make_manifest(tmp: Path, n_per_group=40):
    rng = np.random.default_rng(0)
    rows = []
    groups = {"original": 0, "Deepfakes": 1, "Face2Face": 1, "NeuralTextures": 1}
    splits = ["train"] * 28 + ["val"] * 6 + ["test"] * 6
    for method, label in groups.items():
        d = tmp / method
        d.mkdir(parents=True)
        for i in range(n_per_group):
            nf = rng.integers(8, MAX_LEN + 2)
            center = 1.0 if label == 1 else -1.0
            emb = (rng.standard_normal((nf, EMBED_DIM)) * 0.5 + center).astype("float32")
            p = d / f"{method}_{i:03d}.npy"
            np.save(p, emb)
            rows.append({"video_id": f"{method}_{i:03d}", "method": method,
                         "label": label, "n_frames": int(nf), "embed_dim": EMBED_DIM,
                         "embedding_path": str(p), "split": splits[i % len(splits)]})
    return pd.DataFrame(rows)


def test_subsample():
    with tempfile.TemporaryDirectory() as d:
        man = _make_manifest(Path(d))
        sub = subsample_stratified(man, 30, seed=0)
        assert len(sub) == 30
        assert sub["label"].nunique() == 2          # mantiene ambas clases
        # n mayor que el total -> devuelve todo
        assert len(subsample_stratified(man, 10_000)) == len(man)
        print("  [OK] subsample_stratified")


def test_learning_curve():
    with tempfile.TemporaryDirectory() as d:
        man = _make_manifest(Path(d))
        device = "cpu"
        df = learning_curve(man, CFG, device, sizes=[20, 40, 80])
        assert list(df["n_videos"]) == [20, 40, 80] or len(df) == 3
        assert df["auc"].between(0, 1).all()
        # Datos separables -> AUC alto en el mayor tamaño
        assert df["auc"].iloc[-1] > 0.7
        print(f"  [OK] learning_curve (AUC: {[round(a,2) for a in df['auc']]})")


def test_per_method():
    with tempfile.TemporaryDirectory() as d:
        man = _make_manifest(Path(d))
        device = "cpu"
        # Entrenar un modelo sobre todo
        from src.data.sequence_dataset import get_splits
        parts = get_splits(man)
        model, _, _ = train_eval_hybrid(parts, EMBED_DIM, CFG, device)
        table = per_method_metrics(man, model, CFG, device)
        # Una fila por método presente en test
        assert set(table.index).issubset({"Deepfakes", "Face2Face", "NeuralTextures"})
        assert len(table) >= 1
        assert "auc" in table.columns
        print(f"  [OK] per_method_metrics (métodos: {list(table.index)})")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Ejecutando pruebas de experimentos (Fase 5):")
    test_subsample()
    test_learning_curve()
    test_per_method()
    print("TODAS LAS PRUEBAS DE EXPERIMENTOS PASARON.")
