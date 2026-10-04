"""API REST FastAPI para medir truchas a partir de una foto.

Arranque local::

    uvicorn main:app --reload --host 0.0.0.0 --port 8000

- POST /measure-by-path   JSON {image_path, pixels_per_cm?}
- POST /measure-upload    multipart file + pixels_per_cm opcional
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from config import DEFAULT_MODEL_PATH, OUTPUT_DIR, UPLOADS_DIR
from exceptions import FishNotDetectedError, ImageNotFoundError, InvalidImageError, TroutMeasurerError
from trout_measurer import TroutMeasurer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_measurer: TroutMeasurer | None = None


def get_measurer() -> TroutMeasurer:
    if _measurer is None:
        raise RuntimeError("El modelo YOLO aún no está cargado.")
    return _measurer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carga YOLO una sola vez al arrancar el proceso."""
    global _measurer
    logger.info("Inicializando TroutMeasurer (%s)...", DEFAULT_MODEL_PATH)
    _measurer = TroutMeasurer(model_path=DEFAULT_MODEL_PATH)
    yield
    _measurer = None


app = FastAPI(
    title="Trout Measurer API",
    description="MVP: estima la longitud de una trucha sobre una regla de pesca.",
    version="0.1.0",
    lifespan=lifespan,
)


class MeasureByPathRequest(BaseModel):
    image_path: str = Field(..., description="Ruta local absoluta o relativa a la foto.")
    pixels_per_cm: float | None = Field(
        default=None,
        gt=0,
        description="Calibración manual. Si se omite, se intenta detectar la regla.",
    )

    @field_validator("image_path")
    @classmethod
    def path_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("image_path no puede estar vacío.")
        return value.strip()


class MeasurementResponse(BaseModel):
    success: bool
    length_cm: float
    confidence: float
    pixels_measured: float
    pixels_per_cm_used: float
    processed_image_path: str
    detection_source: str | None = None
    scale_source: str | None = None


def _to_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ImageNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, FishNotDetectedError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (InvalidImageError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, TroutMeasurerError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("Error no controlado en medición.")
    return HTTPException(status_code=500, detail="Error interno al medir la trucha.")


def _decode_upload(data: bytes, filename: str) -> np.ndarray:
    if not data:
        raise InvalidImageError("El archivo subido está vacío.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(f"No se pudo decodificar '{filename}' como imagen.")
    return image


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/measure-by-path", response_model=MeasurementResponse)
def measure_by_path(payload: MeasureByPathRequest) -> MeasurementResponse:
    """Mide una foto que ya está en disco del servidor."""
    try:
        result = get_measurer().measure(
            image_path=payload.image_path,
            pixels_per_cm=payload.pixels_per_cm,
            output_dir=OUTPUT_DIR,
        )
    except Exception as exc:  # noqa: BLE001 — se clasifica en _to_http_error
        raise _to_http_error(exc) from exc

    body = result.to_api_dict()
    body["detection_source"] = result.detection_source
    body["scale_source"] = result.scale_source
    return MeasurementResponse(**body)


@app.post("/measure-upload", response_model=MeasurementResponse)
async def measure_upload(
    file: Annotated[UploadFile, File(description="Imagen JPG/PNG de la trucha sobre la regla.")],
    pixels_per_cm: Annotated[float | None, Form()] = None,
) -> MeasurementResponse:
    """Mide una imagen subida en multipart/form-data."""
    if pixels_per_cm is not None and pixels_per_cm <= 0:
        raise HTTPException(status_code=400, detail="pixels_per_cm debe ser un número positivo.")
    raw = await file.read()
    original_name = Path(file.filename or "upload.jpg").stem or "upload"
    try:
        image = _decode_upload(raw, file.filename or "upload")
        # Persistimos la subida para trazabilidad y para nombrar el overlay.
        saved = UPLOADS_DIR / f"{original_name}{Path(file.filename or 'upload.jpg').suffix or '.jpg'}"
        saved.write_bytes(raw)
        result = get_measurer().measure(
            image_path=saved,
            pixels_per_cm=pixels_per_cm,
            output_dir=OUTPUT_DIR,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_error(exc) from exc

    body = result.to_api_dict()
    body["detection_source"] = result.detection_source
    body["scale_source"] = result.scale_source
    return MeasurementResponse(**body)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
