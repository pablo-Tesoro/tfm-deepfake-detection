"""Pruebas de la Fase 2 sobre tensores/datos sintéticos (requiere torch + sklearn)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.sequence_dataset import (
    SequenceDataset, collate_sequences, split_cross_manipulation, class_weights,
    split_stratified, splits_usable, get_splits,
)
from src.models.baseline import FrameMeanBaseline
from src.models.hybrid import CNNLSTM
from src.training.trainer import train_model
from src.evaluation.metrics import (
    compute_metrics, expected_cost, choose_threshold_by_cost, confusion_counts,
)

EMBED_DIM = 32
MAX_LEN = 16


def _make_synthetic_manifest(tmp: Path, n_per_group: int = 20):
    """Crea embeddings .npy sintéticos separables (reales vs fakes) + manifiesto."""
    rng = np.random.default_rng(0)
    rows = []
    proc = tmp / "processed"
    methods = {"original": 0, "Deepfakes": 1, "Face2Face": 1, "NeuralTextures": 1}
    splits = ["train"] * 14 + ["val"] * 3 + ["test"] * 3  # por grupo
    for method, label in methods.items():
        d = proc / method
        d.mkdir(parents=True)
        for i in range(n_per_group):
            n_frames = rng.integers(8, MAX_LEN + 4)  # longitudes variables
            center = 1.0 if label == 1 else -1.0      # señal separable
            emb = (rng.standard_normal((n_frames, EMBED_DIM)) * 0.5 + center).astype("float32")
            p = d / f"{method}_{i:03d}.npy"
            np.save(p, emb)
            rows.append({
                "video_id": f"{method}_{i:03d}", "method": method, "label": label,
                "n_frames": int(n_frames), "embed_dim": EMBED_DIM,
                "embedding_path": str(p), "split": splits[i % len(splits)],
            })
    return pd.DataFrame(rows)


def test_dataset_and_collate():
    with tempfile.TemporaryDirectory() as d:
        manifest = _make_synthetic_manifest(Path(d))
        ds = SequenceDataset(manifest, max_len=MAX_LEN)
        x, length, y = ds[0]
        assert x.shape == (MAX_LEN, EMBED_DIM)
        assert 1 <= int(length) <= MAX_LEN
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_sequences)
        xb, lengths, ys = next(iter(loader))
        assert xb.shape == (8, MAX_LEN, EMBED_DIM)
        assert lengths.shape == (8,) and ys.shape == (8,)
        print("  [OK] SequenceDataset + collate (formas correctas)")


def test_cross_manipulation_split():
    with tempfile.TemporaryDirectory() as d:
        manifest = _make_synthetic_manifest(Path(d))
        splits = split_cross_manipulation(
            manifest, train_methods=["Deepfakes", "Face2Face"],
            holdout_method="NeuralTextures",
        )
        # El método holdout NO debe aparecer en train ni val
        assert "NeuralTextures" not in set(splits["train"]["method"])
        assert "NeuralTextures" not in set(splits["val"]["method"])
        # En test, los fakes deben ser SOLO del método holdout
        test_fakes = splits["test"][splits["test"]["label"] == 1]
        assert set(test_fakes["method"]) == {"NeuralTextures"}
        # Debe haber reales en los tres conjuntos
        for part in ("train", "val", "test"):
            assert (splits[part]["label"] == 0).sum() > 0
        print("  [OK] split_cross_manipulation (holdout aislado en test)")


def test_models_forward():
    B, T = 4, MAX_LEN
    x = torch.randn(B, T, EMBED_DIM)
    lengths = torch.tensor([T, T - 2, 5, 1], dtype=torch.long)

    base = FrameMeanBaseline(EMBED_DIM, hidden=16)
    out = base(x, lengths)
    assert out.shape == (B,), f"baseline out {out.shape}"

    for rnn in ("lstm", "gru"):
        for bidir in (False, True):
            model = CNNLSTM(EMBED_DIM, hidden=16, rnn_type=rnn, bidirectional=bidir)
            out = model(x, lengths)
            assert out.shape == (B,), f"CNNLSTM {rnn} bidir={bidir} out {out.shape}"
    print("  [OK] forward de FrameMeanBaseline y CNNLSTM (lstm/gru, uni/bi)")


def test_training_reduces_and_learns():
    with tempfile.TemporaryDirectory() as d:
        manifest = _make_synthetic_manifest(Path(d), n_per_group=30)
        train = manifest[manifest["split"] == "train"]
        val = manifest[manifest["split"] == "val"]
        tr = DataLoader(SequenceDataset(train, MAX_LEN), batch_size=16,
                        shuffle=True, collate_fn=collate_sequences)
        va = DataLoader(SequenceDataset(val, MAX_LEN), batch_size=16,
                        collate_fn=collate_sequences)

        model = CNNLSTM(EMBED_DIM, hidden=16)
        out = train_model(model, tr, va, epochs=6, lr=1e-3,
                          pos_weight=class_weights(train), patience=5, verbose=False)
        # Sobre datos separables, el AUC de validación debe ser claramente > 0.5
        assert out["best_val_auc"] > 0.7, f"AUC val={out['best_val_auc']:.3f}"
        print(f"  [OK] train_model aprende (AUC val={out['best_val_auc']:.3f})")


def test_business_metrics():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.4, 0.9, 0.8, 0.35, 0.2, 0.6, 0.45])

    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert set(["accuracy", "precision", "recall", "f1", "auc"]).issubset(m)

    # Si el FN es mucho más caro que el FP, el umbral óptimo debe bajar
    # (clasificar como 'fake' más fácilmente para no dejar pasar deepfakes).
    cheap_fn = choose_threshold_by_cost(y_true, y_prob, cost_fp=1, cost_fn=1)
    costly_fn = choose_threshold_by_cost(y_true, y_prob, cost_fp=1, cost_fn=20)
    assert costly_fn["threshold"] <= cheap_fn["threshold"]
    print(f"  [OK] métricas de negocio (umbral baja con FN caro: "
          f"{cheap_fn['threshold']:.2f} -> {costly_fn['threshold']:.2f})")


def test_stratified_and_auto_splits():
    with tempfile.TemporaryDirectory() as d:
        manifest = _make_synthetic_manifest(Path(d))

        # Estratificado: los tres conjuntos con ambas clases
        strat = split_stratified(manifest, val_frac=0.15, test_frac=0.15)
        assert splits_usable(strat)
        for name in ("train", "val", "test"):
            assert strat[name]["label"].nunique() == 2

        # get_splits con split oficial degenerado (todo a train) -> cae a estratificado
        bad = manifest.copy()
        bad["split"] = "train"          # test y val vacíos en el "oficial"
        parts = get_splits(bad)
        assert splits_usable(parts), "get_splits debería caer a estratificado"
        print("  [OK] split_stratified + get_splits (fallback automático)")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Ejecutando pruebas de la Fase 2 (torch + sklearn):")
    test_dataset_and_collate()
    test_cross_manipulation_split()
    test_stratified_and_auto_splits()
    test_models_forward()
    test_training_reduces_and_learns()
    test_business_metrics()
    print("TODAS LAS PRUEBAS DE FASE 2 PASARON.")
