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

    # Seleccionar especie con advertencia de dominio explícita
    app.selectbox[0].set_value("tomate_arbol").run()
    assert len(app.exception) == 0
    assert any("tomate_arbol" in w.value for w in app.warning)


def _inspection_app(models_dir: Path | None = None, selected_species: str = "tomate_arbol") -> AppTest:
    kwargs = f"selected_species={selected_species!r}"
    if models_dir is not None:
        kwargs = f"models_dir=Path({str(models_dir)!r}), " + kwargs
    code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from pathlib import Path\n"
        "from PIL import Image, ImageDraw\n"
        f"spec = importlib.util.spec_from_file_location('app_mod', {str(PROJECT_ROOT / '06_despliegue/app.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.load_evaluation_summary() is not None\n"
        "im = Image.new('RGB', (240, 240), (245, 245, 245))\n"
        "ImageDraw.Draw(im).ellipse((60, 50, 180, 190), fill=(230, 130, 20))\n"
        f"mod.inspect_fruit_image(im, {kwargs})\n"
    )
    return AppTest.from_string(code, default_timeout=45).run()


def test_streamlit_inspection_flow_on_synthetic_fruit() -> None:
    """Comprueba el flujo completo de inspección (calidad + tamaño + especie) en AppTest."""
    if not (PROJECT_ROOT / "06_despliegue/models/best_quality_model.pt").is_file():
        import pytest

        pytest.skip("Checkpoint local no incluido en el repositorio.")
    app = _inspection_app()
    assert len(app.exception) == 0
    metrics = {m.label: m.value for m in app.metric}
    assert metrics.get("Calidad visual") in QUALITY_CLASSES
    assert metrics.get("Tamaño relativo") not in (None, "No estimado")
    assert any("tomate_arbol" in c.value for c in app.caption)


def test_streamlit_inspection_without_model_preserves_size(tmp_path: Path) -> None:
    """Verifica degradación elegante cuando falta el checkpoint sin perder estimación de tamaño."""
    app = _inspection_app(tmp_path)
    assert len(app.exception) == 0
    metrics = {m.label: m.value for m in app.metric}
    assert metrics.get("Calidad visual") == "Sin modelo"
    assert metrics.get("Puntuación del modelo") == "N/A"
    assert metrics.get("Tamaño relativo") not in (None, "No estimado")
    assert any("No se encontró un modelo" in x.value for x in app.warning)


def test_streamlit_inspection_flow_with_roi_viewfinder() -> None:
    """Verifica el flujo completo con visor puntual ROI activo en ``inspect_fruit_image``."""
    code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from PIL import Image, ImageDraw\n"
        f"spec = importlib.util.spec_from_file_location('app_mod', {str(PROJECT_ROOT / '06_despliegue/app.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "im = Image.new('RGB', (400, 240), (235, 235, 235))\n"
        "draw = ImageDraw.Draw(im)\n"
        "draw.rectangle((270, 15, 385, 230), fill=(140, 80, 35))\n"
        "draw.ellipse((90, 60, 210, 180), fill=(235, 190, 25))\n"
        "mod.inspect_fruit_image(im, selected_species='naranja', roi_params=(38.0, 50.0, 48.0, 75.0))\n"
    )
    app = AppTest.from_string(code, default_timeout=45).run()
    assert len(app.exception) == 0
    metrics = {m.label: m.value for m in app.metric}
    assert "Calidad visual" in metrics
    assert metrics.get("Tamaño relativo") not in (None, "No estimado")
