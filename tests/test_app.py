"""Prueba funcional de la interfaz Streamlit de despliegue (Fase 6 CRISP-DM)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import QUALITY_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_renders_without_exceptions() -> None:
    """Verifica que ``06_despliegue/app.py`` inicia limpiamente sin errores."""
    app = AppTest.from_file(str(PROJECT_ROOT / "06_despliegue/app.py"), default_timeout=25).run()
    assert len(app.exception) == 0
    assert len(app.title) >= 1
    assert "Calidad" in app.title[0].value


def _inspection_app(models_dir: Path | None = None) -> AppTest:
    code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from PIL import Image, ImageDraw\n"
        "from pathlib import Path\n"
        f"spec = importlib.util.spec_from_file_location('app_mod', {str(PROJECT_ROOT / '06_despliegue/app.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "im = Image.new('RGB', (240, 240), (245, 245, 245))\n"
        "ImageDraw.Draw(im).ellipse((60, 50, 180, 190), fill=(230, 130, 20))\n"
        + (f"mod.inspect_fruit_image(im, models_dir=Path({str(models_dir)!r}))\n" if models_dir is not None else "mod.inspect_fruit_image(im)\n")
    )
    return AppTest.from_string(code, default_timeout=45).run()


def test_streamlit_inspection_flow_on_synthetic_fruit() -> None:
    """Smoke test con el checkpoint del equipo, si está instalado."""
    if not (PROJECT_ROOT / "06_despliegue/models/best_quality_model.pt").is_file():
        import pytest
        pytest.skip("Checkpoint local no incluido en el repositorio.")
    app = _inspection_app()
    assert len(app.exception) == 0
    metrics = {m.label: m.value for m in app.metric}
    assert metrics.get("Calidad visual") in QUALITY_CLASSES
    assert metrics.get("Tamaño relativo") not in (None, "No estimado")


def test_streamlit_inspection_without_model_preserves_size(tmp_path: Path) -> None:
    app = _inspection_app(tmp_path)
    assert len(app.exception) == 0
    metrics = {m.label: m.value for m in app.metric}
    assert metrics.get("Calidad visual") == "Sin modelo"
    assert metrics.get("Puntuación del modelo") == "N/A"
    assert metrics.get("Tamaño relativo") not in (None, "No estimado")
    assert any("No se encontró un modelo" in x.value for x in app.warning)
