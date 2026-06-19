"""
Resolución de rutas del proyecto (consciente de local vs. Colab/Drive).

Las SALIDAS (datos, figuras, modelos) deben persistir. En local viven dentro
del proyecto; en Google Colab deben ir a Google Drive, porque el disco de Colab
es efímero y se borra al cerrar la sesión.

Prioridad para decidir la raíz de salidas (workspace):
    1. Variable de entorno TFM_WORKSPACE  (la fija el notebook de Colab -> Drive)
    2. cfg["paths"]["workspace_root"]      (si se define en config.yaml)
    3. La carpeta del proyecto             (comportamiento local por defecto)

Subrutas estándar bajo el workspace:
    data/raw, data/interim, data/processed, reports/figures, models
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def resolve_workspace(cfg: Dict[str, Any], project_root: str | Path) -> Path:
    """Determina la raíz de salidas según la prioridad documentada."""
    env = os.environ.get("TFM_WORKSPACE")
    if env:
        return Path(env)

    cfg_ws = cfg.get("paths", {}).get("workspace_root")
    if cfg_ws:
        return Path(cfg_ws)

    return Path(project_root)


def get_paths(cfg: Dict[str, Any], project_root: str | Path = ".") -> Dict[str, Path]:
    """Devuelve todas las rutas del proyecto ya resueltas (absolutas-relativas).

    Args:
        cfg: configuración cargada (load_config).
        project_root: raíz del proyecto (donde está config/, src/...).

    Returns:
        Diccionario con: project_root, workspace, raw, interim, processed,
        figures, models.
    """
    project_root = Path(project_root)
    ws = resolve_workspace(cfg, project_root)

    return {
        "project_root": project_root,
        "workspace": ws,
        "raw": ws / "data" / "raw",
        "interim": ws / "data" / "interim",
        "processed": ws / "data" / "processed",
        "figures": ws / "reports" / "figures",
        "models": ws / "models",
    }


def ensure_dirs(paths: Dict[str, Path]) -> None:
    """Crea las carpetas de salida si no existen."""
    for key in ("raw", "interim", "processed", "figures", "models"):
        paths[key].mkdir(parents=True, exist_ok=True)
