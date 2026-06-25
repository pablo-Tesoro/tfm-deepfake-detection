"""
VERIFAKE — Consola de verificación forense de autenticidad facial.

App de productivización (Fase 4). Recibe un vídeo, ejecuta el pipeline completo
del TFM (muestreo de frames -> rostro MTCNN -> embeddings -> modelo híbrido
CNN+LSTM) y devuelve:
  - un veredicto AUTÉNTICO / MANIPULADO con su nivel de confianza,
  - una decisión operativa de negocio (aprobar / revisar / rechazar, caso KYC),
  - la "evidencia forense": mapas de calor Grad-CAM sobre los fotogramas,
  - la evolución temporal de la probabilidad de manipulación.

Lanzar (en Colab, con enlace público para la demo y el vídeo):
    from app.app import build_demo
    build_demo().launch(share=True)

Requiere los modelos entrenados en el workspace (models/hybrid.pt, baseline.pt).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Permitir importar el paquete src tanto en local como en Colab
_ROOT = Path(os.environ.get("TFM_PROJECT_ROOT",
                            Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Identidad visual (paleta "consola forense")
# ---------------------------------------------------------------------------
BG = "#0A0F1C"
PANEL = "#101829"
BORDER = "#1E2A44"
TEXT = "#E8EEF7"
MUTED = "#7C8AA8"
REAL = "#1FD1A5"     # autentico (teal)
FAKE = "#FF4D67"     # manipulado (coral)
REVIEW = "#FFB020"   # revision (ambar)
SCAN = "#36CFE0"     # acento cian

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

.gradio-container {{
    background: radial-gradient(1200px 600px at 50% -10%, #122042 0%, {BG} 55%) !important;
    font-family: 'Sora', sans-serif !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
}}
.vf-header {{
    text-align: center; padding: 26px 0 10px;
}}
.vf-logo {{
    font-family: 'Sora', sans-serif; font-weight: 800; letter-spacing: .14em;
    font-size: 30px; color: {TEXT};
}}
.vf-logo .dot {{ color: {SCAN}; }}
.vf-tag {{
    font-family: 'JetBrains Mono', monospace; color: {MUTED};
    font-size: 13px; letter-spacing: .05em; margin-top: 6px;
}}
.vf-panel {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 18px;
}}
.vf-eyebrow {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: .22em; text-transform: uppercase; color: {MUTED};
    margin-bottom: 12px;
}}
.vf-verdict {{
    border-radius: 16px; padding: 22px 24px; border: 1px solid {BORDER};
    display: flex; align-items: center; gap: 18px;
}}
.vf-verdict .word {{
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 30px;
    letter-spacing: .04em; line-height: 1;
}}
.vf-verdict .sub {{
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: {MUTED};
    margin-top: 6px;
}}
.vf-chip {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
    letter-spacing: .12em; padding: 5px 10px; border-radius: 999px;
    border: 1px solid currentColor; display: inline-block;
}}
.vf-kyc {{ border-radius: 14px; padding: 16px 18px; border: 1px solid {BORDER};
    font-family: 'Sora', sans-serif; }}
.vf-kyc .label {{ font-family:'JetBrains Mono',monospace; font-size:11px;
    letter-spacing:.18em; color:{MUTED}; text-transform:uppercase; }}
.vf-kyc .action {{ font-weight:700; font-size:20px; margin-top:4px; }}
.vf-kyc .why {{ color:{MUTED}; font-size:13px; margin-top:6px; }}
button.vf-cta {{
    background: linear-gradient(135deg, {SCAN}, {REAL}) !important;
    color: #042027 !important; font-weight: 700 !important;
    border: none !important; letter-spacing: .03em;
}}
.vf-foot {{ text-align:center; color:{MUTED}; font-family:'JetBrains Mono',monospace;
    font-size:11px; padding:18px 0 6px; line-height:1.7; }}
"""


# ---------------------------------------------------------------------------
# Componentes HTML (anillo de confianza = elemento protagonista)
# ---------------------------------------------------------------------------
def confidence_ring_html(fake_prob: float, threshold: float) -> str:
    """Anillo SVG con el nivel de confianza, coloreado por veredicto."""
    is_fake = fake_prob >= threshold
    conf = fake_prob if is_fake else (1.0 - fake_prob)
    color = FAKE if is_fake else REAL
    label = "MANIPULADO" if is_fake else "AUTÉNTICO"
    pct = int(round(conf * 100))

    r = 92
    circ = 2 * math.pi * r
    offset = circ * (1 - conf)
    return f"""
    <div style="display:flex;justify-content:center;padding:6px 0;">
      <svg width="230" height="230" viewBox="0 0 230 230">
        <circle cx="115" cy="115" r="{r}" fill="none" stroke="{BORDER}" stroke-width="14"/>
        <circle cx="115" cy="115" r="{r}" fill="none" stroke="{color}" stroke-width="14"
            stroke-linecap="round" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}"
            transform="rotate(-90 115 115)" style="filter:drop-shadow(0 0 8px {color}88);"/>
        <text x="115" y="104" text-anchor="middle" fill="{TEXT}"
            style="font-family:'Sora';font-weight:800;font-size:46px;">{pct}%</text>
        <text x="115" y="132" text-anchor="middle" fill="{MUTED}"
            style="font-family:'JetBrains Mono';font-size:12px;letter-spacing:.12em;">CONFIANZA</text>
        <text x="115" y="158" text-anchor="middle" fill="{color}"
            style="font-family:'Sora';font-weight:700;font-size:15px;letter-spacing:.14em;">{label}</text>
      </svg>
    </div>
    """


def verdict_banner_html(fake_prob: float, threshold: float, n_faces: int) -> str:
    is_fake = fake_prob >= threshold
    color = FAKE if is_fake else REAL
    word = "MANIPULADO" if is_fake else "AUTÉNTICO"
    icon = "⚠" if is_fake else "✓"
    sub = (f"Se han detectado patrones sintéticos en el rostro"
           if is_fake else
           f"No se han detectado indicios de manipulación")
    glow = f"box-shadow: inset 0 0 0 1px {color}55, 0 0 40px {color}22;"
    return f"""
    <div class="vf-verdict" style="background:linear-gradient(135deg,{color}1A,{PANEL});{glow}">
      <div style="font-size:40px;color:{color};line-height:1;">{icon}</div>
      <div style="flex:1;">
        <div class="word" style="color:{color};">{word}</div>
        <div class="sub">{sub} · {n_faces} fotogramas analizados · prob. manipulación {fake_prob:.0%}</div>
      </div>
      <div class="vf-chip" style="color:{color};">P = {fake_prob:.2f}</div>
    </div>
    """


def kyc_decision_html(fake_prob: float, low: float = 0.30, high: float = 0.70) -> str:
    """Traduce la probabilidad a una decisión operativa (caso onboarding bancario)."""
    if fake_prob < low:
        color, action, why = (REAL, "APROBAR ALTA",
                              "Riesgo de suplantación bajo. El cliente puede continuar el onboarding.")
    elif fake_prob >= high:
        color, action, why = (FAKE, "RECHAZAR / BLOQUEAR",
                              "Alta probabilidad de deepfake. Se bloquea el alta y se deriva a fraude.")
    else:
        color, action, why = (REVIEW, "REVISIÓN MANUAL",
                              "Resultado en zona de incertidumbre. Se envía a verificación por un agente.")
    return f"""
    <div class="vf-kyc" style="background:linear-gradient(135deg,{color}14,{PANEL});">
      <div class="label">Decisión · Onboarding KYC</div>
      <div class="action" style="color:{color};">{action}</div>
      <div class="why">{why}</div>
    </div>
    """


def _placeholder(text: str) -> str:
    return f"""
    <div class="vf-panel" style="text-align:center;color:{MUTED};
        font-family:'JetBrains Mono',monospace;font-size:13px;padding:40px 18px;">
      {text}
    </div>
    """


def temporal_figure(per_frame_probs: List[float], threshold: float):
    """Curva de probabilidad de manipulación a lo largo del clip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.1), facecolor=BG)
    ax.set_facecolor(PANEL)
    xs = list(range(len(per_frame_probs)))
    ax.plot(xs, per_frame_probs, color=SCAN, linewidth=2.4, marker="o",
            markersize=5, markerfacecolor=SCAN)
    ax.fill_between(xs, per_frame_probs, threshold,
                    where=[p >= threshold for p in per_frame_probs],
                    color=FAKE, alpha=0.18, interpolate=True)
    ax.axhline(threshold, ls="--", color=MUTED, linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("fotograma", color=MUTED, fontsize=10)
    ax.set_ylabel("prob. manipulación", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Motor de detección (carga perezosa de modelos)
# ---------------------------------------------------------------------------
class DeepfakeDetector:
    """Carga los modelos una vez y analiza vídeos."""

    def __init__(self):
        self._ready = False
        self.threshold = 0.5

    def load(self):
        if self._ready:
            return
        import torch
        from src.utils.seeds import load_config
        from src.utils.paths import get_paths
        from src.data.face_extraction import build_mtcnn
        from src.models.hybrid import CNNLSTM
        from src.explainability.gradcam import load_frame_scorer, GradCAM

        cfg = load_config(_ROOT / "config" / "config.yaml")
        paths = get_paths(cfg, _ROOT)
        self.cfg, self.paths = cfg, paths
        self.n_frames = cfg["face_extraction"]["frames_per_video"]
        thr = cfg.get("business", {}).get("decision_threshold")
        self.threshold = float(thr) if thr is not None else 0.5

        # Backbone + cabeza (frame scorer) compartidos para embeddings y Grad-CAM
        self.scorer, self.transform, target_layer, self.device = load_frame_scorer(
            cfg["model"]["backbone"], paths["models"] / "baseline.pt")
        self.backbone = self.scorer.backbone
        self.gradcam = GradCAM(self.scorer, target_layer)
        embed_dim = self.backbone.num_features

        # Modelo híbrido temporal (decisión principal)
        self.hybrid = CNNLSTM(
            embed_dim, hidden=cfg["model"]["hidden_size"],
            num_layers=cfg["model"]["num_layers"],
            rnn_type=cfg["model"]["temporal"], dropout=cfg["model"]["dropout"])
        state = torch.load(paths["models"] / "hybrid.pt", map_location=self.device)
        self.hybrid.load_state_dict(state)
        self.hybrid.to(self.device).eval()

        self.mtcnn = build_mtcnn(
            image_size=cfg["face_extraction"]["image_size"],
            margin=cfg["face_extraction"]["margin"], device=self.device)
        self._ready = True

    def _extract_faces(self, video_path: str):
        """Devuelve (face_imgs_uint8, face_tensors)."""
        from PIL import Image
        from src.data.sampling import sample_frames

        frames = sample_frames(video_path, num_frames=self.n_frames, as_rgb=True)
        face_imgs, face_tensors = [], []
        for frame in frames:
            ft = self.mtcnn(Image.fromarray(frame))
            if ft is None:
                continue
            img = ft.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
            face_imgs.append(img)
            face_tensors.append(self.transform(Image.fromarray(img)))
        return face_imgs, face_tensors

    def analyze(self, video_path: Optional[str]):
        """Pipeline completo. Devuelve (verdict, ring, kyc, gallery, temporal_fig)."""
        import torch
        from src.explainability.gradcam import overlay_cam_on_image

        if not video_path:
            ph = _placeholder("Sube un vídeo y pulsa «Analizar» para comenzar.")
            return ph, _placeholder("· · ·"), "", [], None

        self.load()
        face_imgs, face_tensors = self._extract_faces(video_path)

        if len(face_tensors) < 2:
            msg = ("No se ha podido detectar un rostro con claridad en el vídeo. "
                   "Prueba con un clip donde la cara aparezca de frente y bien iluminada.")
            return (verdict_warning_html(msg), _placeholder("Sin lectura"),
                    "", [], None)

        # Embeddings de la secuencia
        batch = torch.stack(face_tensors).to(self.device)
        with torch.no_grad():
            emb = self.backbone(batch)                       # [n, D]
            seq = emb.unsqueeze(0)                            # [1, n, D]
            lengths = torch.tensor([emb.shape[0]], device=self.device)
            fake_prob = torch.sigmoid(self.hybrid(seq, lengths))[0].item()

        # Por fotograma: probabilidad + Grad-CAM
        per_frame, gallery = [], []
        for i, ft in enumerate(face_tensors):
            x = ft.unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = torch.sigmoid(self.scorer(x))[0, 0].item()
            per_frame.append(p)
            cam = self.gradcam(x)[0]
            overlay = overlay_cam_on_image(face_imgs[i], cam)
            gallery.append((overlay, f"frame {i} · {p:.0%}"))

        verdict = verdict_banner_html(fake_prob, self.threshold, len(face_tensors))
        ring = confidence_ring_html(fake_prob, self.threshold)
        kyc = kyc_decision_html(fake_prob)
        fig = temporal_figure(per_frame, self.threshold)
        return verdict, ring, kyc, gallery, fig


def verdict_warning_html(msg: str) -> str:
    return f"""
    <div class="vf-verdict" style="background:linear-gradient(135deg,{REVIEW}1A,{PANEL});
        box-shadow: inset 0 0 0 1px {REVIEW}55;">
      <div style="font-size:36px;color:{REVIEW};">⚠</div>
      <div style="flex:1;"><div class="word" style="color:{REVIEW};font-size:22px;">SIN LECTURA</div>
      <div class="sub">{msg}</div></div>
    </div>
    """


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
def build_demo():
    import gradio as gr

    detector = DeepfakeDetector()
    theme = gr.themes.Base(
        primary_hue="teal", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Sora"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(body_background_fill=BG, body_text_color=TEXT, block_background_fill=PANEL)

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="VERIFAKE") as demo:
        gr.HTML(f"""
        <div class="vf-header">
          <div class="vf-logo">VERI<span class="dot">·</span>FAKE</div>
          <div class="vf-tag">ANÁLISIS FORENSE DE AUTENTICIDAD FACIAL · ONBOARDING DIGITAL</div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                gr.HTML('<div class="vf-eyebrow">Entrada · vídeo de verificación</div>')
                video_in = gr.Video(label=None, height=300)
                analyze_btn = gr.Button("Analizar autenticidad", elem_classes="vf-cta", size="lg")
                gr.HTML(f"""<div style="color:{MUTED};font-family:'JetBrains Mono',monospace;
                    font-size:11px;margin-top:8px;line-height:1.7;">
                    Pipeline: muestreo de fotogramas → recorte facial (MTCNN) →
                    embeddings (CNN) → coherencia temporal (LSTM).</div>""")
            with gr.Column(scale=5):
                gr.HTML('<div class="vf-eyebrow">Veredicto</div>')
                verdict_out = gr.HTML(_placeholder("Esperando análisis…"))
                with gr.Row():
                    ring_out = gr.HTML(_placeholder("· · ·"))
                kyc_out = gr.HTML("")

        gr.HTML('<div class="vf-eyebrow" style="margin-top:18px;">Evidencia forense · mapas de activación (Grad-CAM)</div>')
        gallery_out = gr.Gallery(label=None, columns=8, height=180, object_fit="cover", show_label=False)

        gr.HTML('<div class="vf-eyebrow" style="margin-top:8px;">Evolución temporal de la sospecha</div>')
        temporal_out = gr.Plot(label=None)

        gr.HTML(f"""<div class="vf-foot">
            VERIFAKE · prototipo académico (TFM, Máster en Ciencia de Datos e IA — UCM)<br>
            Detección de deepfakes mediante arquitectura híbrida CNN+LSTM y explicabilidad Grad-CAM.
            No constituye una herramienta de verificación certificada.
        </div>""")

        analyze_btn.click(
            fn=detector.analyze, inputs=video_in,
            outputs=[verdict_out, ring_out, kyc_out, gallery_out, temporal_out],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(share=True)
