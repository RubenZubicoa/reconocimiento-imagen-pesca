"""Excepciones de dominio para la medición de truchas."""

from __future__ import annotations


class TroutMeasurerError(Exception):
    """Error base del pipeline de medición."""


class ImageNotFoundError(TroutMeasurerError, FileNotFoundError):
    """La ruta de imagen no existe en disco."""


class InvalidImageError(TroutMeasurerError):
    """El archivo existe pero no se pudo decodificar como imagen."""


class FishNotDetectedError(TroutMeasurerError):
    """YOLO (y el fallback OpenCV) no encontraron ningún pez."""
