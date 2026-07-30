"""
run_all.py — Orquestador de extremo a extremo.

Ejecuta TODO el proyecto en una sola llamada y lanza la app, devolviendo la URL:

    splits -> (descarga opcional) -> vídeo->embeddings (fusionado, 1-2 backbones)
    -> entrenamiento (baseline + híbrido) -> evaluación y figuras
    -> experimentos (curva de aprendizaje, backbones, por método) -> app.

Optimizado para Google Drive: no escribe frames intermedios (vídeo -> .npy
directo), detección facial por lotes, un solo escaneo por directorio,
reutilización de manifiestos y caché en RAM de embeddings durante el
entrenamiento. Cada paso es idempotente: lo ya hecho no se repite.

    from run_all import run_pipeline
    run_pipeline()                                   # reutiliza lo ya hecho
    run_pipeline(download=True, n_videos=None, retrain=True)   # todo desde cero
    run_pipeline(experiments=False)                  # sin los 3 experimentos
    run_pipeline(compare_backbone=None)              # sin la 2ª pasada (ResNet)
"""
from __future__ import annotations

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("TFM_PROJECT_ROOT",
                                   Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT))


def _banner(step: int, total: int, text: str) -> None:
    print(f"\n{'='*64}\n[{step}/{total}]  {text}\n{'='*64}")


def _load_context():
    from src.utils.seeds import load_config, set_seed
    from src.utils.paths import get_paths, ensure_dirs
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    set_seed(cfg["project"]["seed"])
    paths = get_paths(cfg, PROJECT_ROOT)
    ensure_dirs(paths)
    return cfg, paths


def ensure_splits(paths) -> None:
    """Descarga los splits oficiales de FF++ si faltan (públicos)."""
    splits_dir = paths["raw"] / "splits"
    names = ["train.json", "val.json", "test.json"]
    if all((splits_dir / n).exists() for n in names):
        print("Splits oficiales: ya presentes.")
        return
    splits_dir.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"
    for n in names:
        if not (splits_dir / n).exists():
            urllib.request.urlretrieve(f"{base}/{n}", splits_dir / n)
            print("  descargado", n)


def run_download(paths, cfg, n_videos: int, ff_script: str) -> None:
    """Ejecuta el script oficial de FF++ SOLO para las categorías que usamos.

    En lugar de '-d all' (que baja 8 categorías, 3 de ellas inútiles para este
    proyecto), se llama una vez por cada dataset necesario: los vídeos reales
    ('original') y los 4 métodos del config. Se alimenta el prompt de términos
    de uso del script con una línea para que no bloquee en automático.
    """
    script = PROJECT_ROOT / ff_script
    if not script.exists():
        print(f"[descarga omitida] No encuentro {script}. "
              f"Coloca el script de FaceForensics++ ahí para descargar automáticamente.")
        return

    datasets = ["original"] + list(cfg["dataset"]["manipulation_methods"])
    alcance = "TODOS los vídeos" if not n_videos else f"{n_videos} por categoría"
    print(f"Descargando solo las categorías usadas ({alcance}): {datasets}")
    for ds in datasets:
        cmd = [sys.executable, str(script), str(paths["raw"]),
               "-d", ds, "-c", cfg["dataset"]["compression"],
               "-t", "videos", "--server", "EU2"]
        if n_videos:                       # None o 0 -> sin límite (todo el dataset)
            cmd += ["-n", str(n_videos)]
        print("  ->", ds)
        # input='\n' acepta el prompt de términos de uso (TOS) sin intervención.
        subprocess.run(cmd, input="\n", text=True, check=False)


def train_models(cfg, paths, manifest, retrain: bool):
    """Entrena (o carga) baseline e híbrido. Devuelve modelos + particiones."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.sequence_dataset import (
        SequenceDataset, collate_sequences, get_splits, class_weights)
    from src.models.baseline import FrameMeanBaseline
    from src.models.hybrid import CNNLSTM
    from src.training.trainer import train_model

    embed_dim = int(manifest["embed_dim"].iloc[0])
    max_len = cfg["face_extraction"]["frames_per_video"]
    parts = get_splits(manifest)
    for k, v in parts.items():
        print(f"  {k:5}: {len(v):4d} vídeos | fakes={int((v['label']==1).sum())}")

    batch = cfg["training"]["batch_size"]
    loaders = {n: DataLoader(SequenceDataset(df, max_len), batch_size=batch,
                             shuffle=(n == "train"), collate_fn=collate_sequences)
               for n, df in parts.items()}
    pos_w = class_weights(parts["train"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline = FrameMeanBaseline(embed_dim, hidden=cfg["model"]["hidden_size"])
    hybrid = CNNLSTM(embed_dim, hidden=cfg["model"]["hidden_size"],
                     num_layers=cfg["model"]["num_layers"],
                     rnn_type=cfg["model"]["temporal"], dropout=cfg["model"]["dropout"])

    ckpt_b = paths["models"] / "baseline.pt"
    ckpt_h = paths["models"] / "hybrid.pt"

    if ckpt_b.exists() and ckpt_h.exists() and not retrain:
        print("Modelos ya entrenados: se cargan (usa retrain=True para reentrenar).")
        baseline.load_state_dict(torch.load(ckpt_b, map_location=device))
        hybrid.load_state_dict(torch.load(ckpt_h, map_location=device))
    else:
        print("Entrenando baseline...")
        train_model(baseline, loaders["train"], loaders["val"],
                    epochs=cfg["training"]["epochs"], lr=cfg["training"]["learning_rate"],
                    pos_weight=pos_w, device=device, checkpoint_path=ckpt_b, verbose=False)
        print("Entrenando híbrido CNN+LSTM...")
        train_model(hybrid, loaders["train"], loaders["val"],
                    epochs=cfg["training"]["epochs"], lr=cfg["training"]["learning_rate"],
                    pos_weight=pos_w, device=device, checkpoint_path=ckpt_h, verbose=False)

    baseline.to(device).eval(); hybrid.to(device).eval()
    return baseline, hybrid, parts, loaders, device


def make_figures(cfg, paths, inventory, manifest, baseline, hybrid, loaders, device):
    """Genera las figuras clave para la memoria (robusto ante fallos)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = paths["figures"]

    # 1) Distribución de clases
    try:
        counts = inventory["label"].map({0: "Real", 1: "Fake"}).value_counts()
        ax = counts.plot(kind="bar", color=["#2a9d8f", "#e76f51"], figsize=(5, 4))
        ax.set_title("Distribución de clases"); ax.set_ylabel("vídeos")
        plt.tight_layout(); plt.savefig(fig_dir / "01_distribucion_clases.png", dpi=120)
        plt.close()
    except Exception as e:
        print("  [fig] distribución:", e)

    # 2) Métricas comparativas + matriz de confusión (en test)
    try:
        from src.training.trainer import _predict_proba
        from src.evaluation.metrics import compute_metrics, metrics_table, plot_confusion
        y, pb = _predict_proba(baseline, loaders["test"], device)
        _, ph = _predict_proba(hybrid, loaders["test"], device)
        table = metrics_table({"Baseline": compute_metrics(y, pb),
                               "Híbrido CNN+LSTM": compute_metrics(y, ph)})
        table.to_csv(fig_dir / "tabla_comparativa.csv")
        print("\nMétricas en test:\n", table)
        plot_confusion(y, ph, 0.5, title="Híbrido — matriz de confusión")
        plt.tight_layout(); plt.savefig(fig_dir / "matriz_confusion.png", dpi=120)
        plt.close()
    except Exception as e:
        print("  [fig] métricas/confusión:", e)

    print("Figuras guardadas en", fig_dir)


def run_experiments(cfg, paths, inv, manifest, hybrid, device,
                    compare_backbone="resnet50", lc_sizes=None):
    """Los 3 experimentos avanzados; cada uno guarda su CSV/figura y es robusto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.experiments.comparisons import (
        learning_curve, plot_learning_curve, compare_backbones, per_method_metrics)

    fig_dir = paths["figures"]

    # 1) Curva de aprendizaje: AUC vs nº de vídeos de entrenamiento
    try:
        print("\n-- Curva de aprendizaje (AUC vs nº de vídeos) --")
        lc = learning_curve(manifest, cfg, device, sizes=lc_sizes)
        lc.to_csv(fig_dir / "curva_aprendizaje.csv", index=False)
        plot_learning_curve(lc, save_path=fig_dir / "curva_aprendizaje.png")
        plt.close("all")
    except Exception as e:
        print("  [exp] curva de aprendizaje:", e)

    # 2) Comparativa de backbones (EfficientNet vs ResNet)
    if compare_backbone:
        try:
            print(f"\n-- Backbones: {cfg['model']['backbone']} vs {compare_backbone} --")
            bt = compare_backbones(inv, paths["processed"], cfg, device,
                                   backbones=[cfg["model"]["backbone"], compare_backbone])
            print(bt)
            bt.to_csv(fig_dir / "comparativa_backbones.csv")
        except Exception as e:
            print("  [exp] backbones:", e)

    # 3) Métricas por método de manipulación (reutiliza el híbrido ya entrenado)
    try:
        print("\n-- Métricas por método de manipulación --")
        pm = per_method_metrics(manifest, hybrid, cfg, device)
        print(pm)
        if not pm.empty:
            pm.to_csv(fig_dir / "metricas_por_metodo.csv")
            ax = pm["auc"].plot(kind="bar", color="#457b9d", figsize=(7, 4))
            ax.set_title("AUC por método de manipulación"); ax.set_ylim(0, 1)
            plt.xticks(rotation=20); plt.tight_layout()
            plt.savefig(fig_dir / "auc_por_metodo.png", dpi=120)
            plt.close("all")
    except Exception as e:
        print("  [exp] por método:", e)

    print("\nResultados de experimentos guardados en", fig_dir)


def run_pipeline(
    download: bool = False,
    n_videos: int = 150,
    ff_script: str = "download-FaceForensics.py",
    retrain: bool = False,
    make_figs: bool = True,
    experiments: bool = True,
    compare_backbone: str | None = "resnet50",
    lc_sizes=None,
    launch: bool = True,
    share: bool = True,
):
    """Ejecuta el proyecto completo y (opcionalmente) lanza la app.

    Args:
        download: descargar FF++ (requiere el script oficial en el repo).
        n_videos: vídeos por categoría al descargar; None = todos.
        retrain: reentrenar aunque existan checkpoints.
        make_figs: generar las figuras básicas para la memoria.
        experiments: ejecutar los 3 experimentos (curva de aprendizaje,
            comparativa de backbones y métricas por método).
        compare_backbone: segundo backbone a comparar ('resnet50'); None lo
            omite y ahorra la pasada extra vídeo->embedding.
        lc_sizes: tamaños de la curva de aprendizaje; None = automático.
        launch/share: lanzar la app Gradio con enlace público.

    Returns:
        El objeto `demo` de Gradio si launch=True (con .share_url), si no None.
    """
    from src.data.dataset import enumerate_videos, load_official_splits, assign_splits
    from src.features.embeddings import embed_videos_multi

    TOTAL = 7
    cfg, paths = _load_context()

    _banner(1, TOTAL, "Splits oficiales")
    ensure_splits(paths)
    if download:
        run_download(paths, cfg, n_videos, ff_script)

    _banner(2, TOTAL, "Inventario de vídeos")
    inv = enumerate_videos(paths["raw"], compression=cfg["dataset"]["compression"],
                           methods=cfg["dataset"]["manipulation_methods"])
    if inv.empty:
        print("No hay vídeos en", paths["raw"],
              "\nDescarga FaceForensics++ (download=True con el script colocado).")
        return None
    splits_dir = paths["raw"] / "splits"
    if splits_dir.exists():
        inv["split"] = assign_splits(inv, load_official_splits(splits_dir))
    print(f"{len(inv)} vídeos ({int((inv['label']==0).sum())} reales / "
          f"{int((inv['label']==1).sum())} fakes).")

    extra = (compare_backbone,) if (experiments and compare_backbone) else ()
    titulo = "Vídeo -> embeddings (fusionado" + (", MTCNN compartido para 2 backbones)" if extra else ")")
    _banner(3, TOTAL, titulo)
    manifests = embed_videos_multi(
        inv, paths["processed"], backbone=cfg["model"]["backbone"],
        extra_backbones=extra,
        num_frames=cfg["face_extraction"]["frames_per_video"],
        image_size=cfg["face_extraction"]["image_size"],
        margin=cfg["face_extraction"]["margin"])
    manifest = manifests[cfg["model"]["backbone"]]
    if manifest.empty:
        print("No se generaron embeddings (¿hay vídeos y se detectan rostros?).")
        return None

    _banner(4, TOTAL, "Entrenamiento (baseline + híbrido)")
    baseline, hybrid, parts, loaders, device = train_models(cfg, paths, manifest, retrain)

    _banner(5, TOTAL, "Evaluación y figuras para la memoria")
    if make_figs:
        make_figures(cfg, paths, inv, manifest, baseline, hybrid, loaders, device)
    else:
        print("(omitidas)")

    _banner(6, TOTAL, "Experimentos avanzados")
    if experiments:
        run_experiments(cfg, paths, inv, manifest, hybrid, device,
                        compare_backbone=compare_backbone, lc_sizes=lc_sizes)
    else:
        print("(omitidos: experiments=False)")

    _banner(7, TOTAL, "Lanzando la app VERIFAKE")
    if not launch:
        print("(launch=False) Pipeline completado.")
        return None
    from app.app import build_demo
    demo = build_demo()
    demo.launch(share=share)
    url = getattr(demo, "share_url", None) or getattr(demo, "local_url", None)
    print("\n" + "★" * 64)
    print("  APP LISTA. URL pública:", url)
    print("★" * 64)
    return demo


if __name__ == "__main__":
    run_pipeline()
