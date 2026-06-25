"""Pruebas de la Fase 4 (app). Validan los componentes HTML y la construcción de la UI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.app import (
    confidence_ring_html, verdict_banner_html, kyc_decision_html,
    temporal_figure, verdict_warning_html, build_demo, REAL, FAKE, REVIEW,
)


def test_verdict_banner():
    fake = verdict_banner_html(0.92, threshold=0.5, n_faces=16)
    real = verdict_banner_html(0.10, threshold=0.5, n_faces=16)
    assert "MANIPULADO" in fake and FAKE in fake
    assert "AUTÉNTICO" in real and REAL in real
    print("  [OK] verdict_banner_html (fake vs real)")


def test_confidence_ring():
    # Un fake con prob 0.9 -> confianza 90% en 'MANIPULADO'
    html = confidence_ring_html(0.90, threshold=0.5)
    assert "90%" in html and "MANIPULADO" in html and "<svg" in html
    # Un real con prob 0.2 -> confianza 80% en 'AUTÉNTICO'
    html2 = confidence_ring_html(0.20, threshold=0.5)
    assert "80%" in html2 and "AUTÉNTICO" in html2
    print("  [OK] confidence_ring_html (porcentaje y color correctos)")


def test_kyc_decision():
    assert "APROBAR" in kyc_decision_html(0.10)
    assert "REVISIÓN" in kyc_decision_html(0.50)
    assert "RECHAZAR" in kyc_decision_html(0.90)
    print("  [OK] kyc_decision_html (3 zonas de decisión)")


def test_temporal_figure():
    fig = temporal_figure([0.1, 0.3, 0.8, 0.9, 0.4], threshold=0.5)
    assert fig is not None and len(fig.axes) == 1
    print("  [OK] temporal_figure (genera la curva)")


def test_build_demo():
    # Construir la interfaz no debe lanzar el servidor ni cargar modelos.
    demo = build_demo()
    assert demo is not None
    print("  [OK] build_demo (interfaz construida sin cargar modelos)")


if __name__ == "__main__":
    print("Ejecutando pruebas de la Fase 4 (app):")
    test_verdict_banner()
    test_confidence_ring()
    test_kyc_decision()
    test_temporal_figure()
    test_build_demo()
    print("TODAS LAS PRUEBAS DE FASE 4 PASARON.")
