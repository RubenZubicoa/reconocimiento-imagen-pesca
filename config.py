"""Configuración central del MVP de medición de truchas."""

from __future__ import annotations

import os
from pathlib import Path

# Directorio raíz del proyecto (donde está este archivo).
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Modelo YOLO
# ---------------------------------------------------------------------------
# V1: peso preentrenado de Ultralytics. COCO no incluye la clase "fish",
# por eso TroutMeasurer caerá a la mayor detección o a un contorno OpenCV
# hasta que se sustituya por un .pt propio (p. ej. models/trout-seg.pt).
DEFAULT_MODEL_PATH: str = os.getenv("TROUT_MODEL_PATH", "yolov8n-seg.pt")

# Umbral de confianza de YOLO (0–1). Bajamos un poco para fotos de campo.
CONFIDENCE_THRESHOLD: float = float(os.getenv("TROUT_CONFIDENCE", "0.25"))

# Nombres de clase que consideramos "pez" cuando el modelo es personalizado.
FISH_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "fish",
        "trout",
        "trucha",
        "salmon",
        "salmo",
        "rainbow trout",
        "brown trout",
    }
)

# ---------------------------------------------------------------------------
# Escala píxeles → centímetros
# ---------------------------------------------------------------------------
# Fallback de laboratorio cuando no se pasa pixels_per_cm ni se detecta regla.
# Interpretación: 10 px equivalen a 1 cm  =>  longitud_cm = d_px / 10.
DEFAULT_PIXELS_PER_CM: float = float(os.getenv("TROUT_PIXELS_PER_CM", "10.0"))

# Distancia real (cm) entre marcas consecutivas de la regla, para el
# estimador automático por ticks. En la mayoría de medidores de pesca es 1 cm.
RULER_TICK_SPACING_CM: float = float(os.getenv("TROUT_TICK_SPACING_CM", "1.0"))

# Longitud real conocida de la regla completa (cm). Si se detecta el bbox
# de un objeto alargado tipo regla, pixels_per_cm ≈ ruler_px / esta longitud.
DEFAULT_RULER_LENGTH_CM: float | None = None

# ---------------------------------------------------------------------------
# Rutas de E/S
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"
MODELS_DIR: Path = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
