"""Pipeline de visión por computador para estimar la longitud de una trucha.

Flujo V1
--------
A. Cargar un modelo YOLO (seg/pose/detect). Por defecto ``yolov8n-seg.pt``.
B. (Opcional) Alinear la foto con una homografía si se conocen las 4 esquinas
   de la regla o se detecta un tablero rectangular.
C. Detectar el pez (máscara, keypoints o bbox) y extraer hocico y cola.
D. Convertir la distancia euclidiana en píxeles a centímetros con una escala
   ``pixels_per_cm`` conocida, detectada en la regla, o de fallback.

Matemática de la conversión
---------------------------
Sea el hocico ``(x1, y1)`` y la cola ``(x2, y2)`` en el plano de la imagen:

    d_px = sqrt( (x2 - x1)^2 + (y2 - y1)^2 )

Si ``s`` es la escala en píxeles por centímetro (pixels_per_cm):

    longitud_cm = d_px / s

Equivalente: cada centímetro real ocupa ``s`` píxeles, luego dividir la
longitud medida en píxeles entre ``s`` recupera centímetros.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_PIXELS_PER_CM,
    DEFAULT_RULER_LENGTH_CM,
    FISH_CLASS_NAMES,
    OUTPUT_DIR,
    RULER_TICK_SPACING_CM,
)
from exceptions import FishNotDetectedError, ImageNotFoundError, InvalidImageError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------


@dataclass
class MeasurementResult:
    """Salida estructurada del medidor (lista para serializar a JSON)."""

    success: bool
    length_cm: float
    confidence: float
    pixels_measured: float
    pixels_per_cm_used: float
    processed_image_path: str
    snout_xy: tuple[float, float]
    tail_xy: tuple[float, float]
    detection_source: str = "yolo"
    scale_source: str = "parameter"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        """Campos del contrato REST pedido en el MVP."""
        return {
            "success": self.success,
            "length_cm": round(self.length_cm, 2),
            "confidence": round(self.confidence, 4),
            "pixels_measured": round(self.pixels_measured, 2),
            "pixels_per_cm_used": round(self.pixels_per_cm_used, 4),
            "processed_image_path": self.processed_image_path,
        }


# ---------------------------------------------------------------------------
# I/O de imagen (Windows + rutas con acentos)
# ---------------------------------------------------------------------------


def read_image(path: str | Path) -> np.ndarray:
    """Lee BGR con ``imdecode`` para no fallar en rutas no ASCII de Windows."""
    path = Path(path)
    if not path.is_file():
        raise ImageNotFoundError(f"No existe la imagen: {path}")

    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(f"No se pudo decodificar la imagen: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> Path:
    """Escribe una imagen BGR creando directorios intermedios si hace falta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise InvalidImageError(f"No se pudo codificar la imagen de salida: {path}")
    encoded.tofile(str(path))
    return path


# ---------------------------------------------------------------------------
# Homografía / alineación (Paso B)
# ---------------------------------------------------------------------------


def order_corners(points: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como (TL, TR, BR, BL) para ``cv2.getPerspectiveTransform``.

    Se usa la suma y la diferencia de coordenadas:
    - TL tiene la menor x+y, BR la mayor x+y.
    - TR tiene la menor y-x (o mayor x-y), BL la mayor y-x.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    return ordered


def warp_with_homography(
    image: np.ndarray,
    corners: Sequence[Sequence[float]],
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectifica el plano de la regla mediante homografía.

    Dados 4 puntos de la regla/tablero en la foto (que en el mundo real forman
    un rectángulo), estimamos la matriz H 3×3 tal que:

        [x']       [x]
        [y'] = H · [y]
        [1 ]       [1]

    Tras el warp, distancias sobre el plano de la regla dejan de estar
    distorsionadas por la perspectiva (en V1 es una corrección básica).

    Returns
    -------
    warped, H
        Imagen alineada y matriz de homografía 3×3.
    """
    src = order_corners(np.asarray(corners, dtype=np.float32))
    width_a = float(np.linalg.norm(src[1] - src[0]))
    width_b = float(np.linalg.norm(src[2] - src[3]))
    height_a = float(np.linalg.norm(src[3] - src[0]))
    height_b = float(np.linalg.norm(src[2] - src[1]))
    width = int(max(width_a, width_b))
    height = int(max(height_a, height_b))
    if output_size is not None:
        width, height = output_size

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, homography, (width, height))
    return warped, homography


def detect_board_corners(image: np.ndarray) -> np.ndarray | None:
    """Intenta hallar el mayor cuadrilátero (tablero o zona de la regla).

    Útil para recortar/alinear el área de interés cuando no se pasan esquinas.
    Devuelve None si no hay un candidato razonable.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    best: np.ndarray | None = None
    best_area = 0.0
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = float(cv2.contourArea(approx))
        # El tablero suele ocupar una fracción grande de la foto, no un recorte minúsculo.
        if area < 0.15 * image_area:
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype(np.float32)
    return best


def crop_interest_region(
    image: np.ndarray,
    margin_ratio: float = 0.02,
) -> np.ndarray:
    """Recorte básico: elimina un pequeño margen o usa el tablero detectado."""
    corners = detect_board_corners(image)
    if corners is not None:
        warped, _ = warp_with_homography(image, corners)
        logger.info("Área de interés alineada por homografía automática.")
        return warped

    h, w = image.shape[:2]
    mx, my = int(w * margin_ratio), int(h * margin_ratio)
    return image[my : h - my, mx : w - mx].copy()


# ---------------------------------------------------------------------------
# Detección de pez (Paso C)
# ---------------------------------------------------------------------------


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """d_px = sqrt((x2 - x1)^2 + (y2 - y1)^2)."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    return float(np.hypot(x2 - x1, y2 - y1))


def extrema_along_principal_axis(
    points_xy: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Hocico y cola como extremos de la máscara proyectados sobre su eje mayor.

    En lugar de usar el píxel más a la izquierda/derecha (falla si el pez está
    en diagonal), se estima el eje principal por PCA (SVD de puntos centrados):

        X_c = X - mean(X)
        X_c = U Σ V^T      →  dirección principal = primera fila de V^T

    Cada punto se proyecta:  t_i = (p_i - mean) · direction
    El min y el max de t_i son cola y hocico (o viceversa; el orden visual
    no afecta a la distancia euclidiana).
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        raise FishNotDetectedError("La máscara del pez no tiene puntos suficientes.")

    mean = pts.mean(axis=0)
    centered = pts - mean
    # PCA 2D: el primer componente es la dirección de máxima varianza (eje del pez).
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projections = centered @ direction
    snout = pts[int(np.argmin(projections))]
    tail = pts[int(np.argmax(projections))]
    return (float(snout[0]), float(snout[1])), (float(tail[0]), float(tail[1]))


def extrema_from_mask(mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Extremos hocico/cola a partir de una máscara binaria (H×W)."""
    binary = (mask > 0.5).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        raise FishNotDetectedError("Máscara vacía: no hay píxeles de pez.")
    points = np.stack([xs, ys], axis=1)
    return extrema_along_principal_axis(points)


def extrema_from_bbox(
    xyxy: Sequence[float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fallback: extremos del eje más largo de la caja axis-aligned.

    Si el pez está más horizontal que vertical, hocico/cola ≈ (x1, cy) y (x2, cy).
    Si está más vertical, se usan (cx, y1) y (cx, y2).
    """
    x1, y1, x2, y2 = map(float, xyxy)
    width, height = x2 - x1, y2 - y1
    if width >= height:
        cy = (y1 + y2) / 2.0
        return (x1, cy), (x2, cy)
    cx = (x1 + x2) / 2.0
    return (cx, y1), (cx, y2)


def extrema_from_keypoints(
    keypoints_xy: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Con pose: se toman los keypoints válidos más extremos sobre el eje PCA."""
    pts = np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2)
    valid = pts[np.isfinite(pts).all(axis=1) & (pts.sum(axis=1) > 0)]
    if len(valid) < 2:
        raise FishNotDetectedError("No hay keypoints suficientes para hocico/cola.")
    return extrema_along_principal_axis(valid)


def _largest_component_mask(binary: np.ndarray) -> np.ndarray | None:
    """Devuelve la máscara del mayor contorno externo, o None."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(binary.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
    return mask


def _mask_looks_like_fish(mask: np.ndarray, image: np.ndarray) -> bool:
    """Filtro rápido: el pez es un blob alargado, no el tablero entero ni una hoja."""
    binary = (mask > 0.5).astype(np.uint8)
    area = float(cv2.countNonZero(binary))
    image_area = float(image.shape[0] * image.shape[1])
    if area < 0.02 * image_area or area > 0.55 * image_area:
        return False
    xs = np.where(binary > 0)[1]
    ys = np.where(binary > 0)[0]
    if len(xs) < 20:
        return False
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    aspect = max(w, h) / max(min(w, h), 1.0)
    return 2.0 <= aspect <= 12.0


def _best_elongated_contour(
    image: np.ndarray,
    binary: np.ndarray,
    hsv: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """Elige el contorno más 'de pez': alargado, central, de tamaño medio."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    if hsv is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image_h, image_w = image.shape[:2]
    image_area = float(image_h * image_w)
    best_mask: np.ndarray | None = None
    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.008 * image_area or area > 0.55 * image_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect < 2.2:
            continue
        thin_bar = min(w, h) < 0.10 * min(image_h, image_w) and aspect > 8.0
        if thin_bar:
            continue

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
        cy = y + h / 2.0
        # Las plantas suelen estar arriba; el pez, en el centro/abajo junto a la regla.
        vertical_prior = 1.0 - abs((cy / image_h) - 0.52)
        sat_mean = float(hsv[:, :, 1][mask > 0].mean()) if np.any(mask) else 0.0
        hue_mean = float(hsv[:, :, 0][mask > 0].mean()) if np.any(mask) else 0.0
        # Penaliza verde saturado de follaje (no el pez plateado/oliva).
        foliage_penalty = 0.35 if (32 <= hue_mean <= 85 and sat_mean > 70 and cy < 0.42 * image_h) else 1.0
        score = (aspect ** 1.3) * float(np.log1p(area)) * max(vertical_prior, 0.2) * foliage_penalty
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        return None
    confidence = float(min(0.75, 0.35 + best_score / 400.0))
    return best_mask, confidence


def find_fish_by_contour(
    image: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Fallback OpenCV: mayor contorno alargado (fotos sintéticas o si GrabCut falla)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, (5, 40, 25), (100, 255, 220))
    sat_mask = cv2.inRange(hsv[:, :, 1], 50, 255)
    combined = cv2.bitwise_or(color_mask, sat_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    picked = _best_elongated_contour(image, combined, hsv=hsv)
    if picked is not None:
        return picked

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    )
    vertical_ticks = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 22))
    )
    dark = cv2.subtract(dark, horizontal)
    dark = cv2.subtract(dark, vertical_ticks)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _best_elongated_contour(image, dark, hsv=hsv)


def detect_yellow_tape_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box de una cinta métrica amarilla (aspecto muy alargado)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tape = cv2.inRange(hsv, (18, 120, 120), (40, 255, 255))
    tape = cv2.morphologyEx(
        tape, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 7))
    )
    tape = cv2.morphologyEx(
        tape, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    )
    contours, _ = cv2.findContours(tape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    image_area = float(image.shape[0] * image.shape[1])
    best: tuple[int, int, int, int] | None = None
    best_score = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < 0.004 * image_area:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect < 6.0:
            continue
        score = area * aspect
        if score > best_score:
            best_score = score
            best = (x, y, w, h)
    return best


def segment_fish_grabcut(
    image: np.ndarray,
    tape_bbox: tuple[int, int, int, int] | None = None,
    target_size: int = 800,
) -> np.ndarray | None:
    """Segmenta el pez con GrabCut a resolución reducida (rápido en fotos 4K).

    Si hay cinta amarilla, el pez se busca pegado a ella (arriba o abajo) y la
    cinta/césped se marcan como fondo. Si no, se asume un medidor oscuro con
    la escala en la franja superior (foto tipo stingwrap).
    """
    h, w = image.shape[:2]
    scale = target_size / float(max(h, w))
    if scale < 1.0:
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = image
        scale = 1.0
    sh, sw = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (18, 120, 120), (40, 255, 255))
    grass = cv2.inRange(hsv, (32, 60, 25), (85, 255, 230))
    grass[int(0.42 * sh) :, :] = 0  # no marcar el cuerpo oliva del pez como césped

    mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)

    if tape_bbox is not None:
        tx, ty, tw, th = [int(round(v * scale)) for v in tape_bbox]
        tx, ty = max(0, tx), max(0, ty)
        tw, th = max(1, tw), max(1, th)
        mask[:, :] = cv2.GC_BGD
        if ty > 0.40 * sh:
            fy0 = max(0, ty - int(0.42 * sh))
            mask[fy0:ty, tx : min(sw, tx + tw)] = cv2.GC_PR_FGD
            mask[ty:, :] = cv2.GC_BGD
        else:
            fy1 = min(sh, ty + th + int(0.80 * sh))
            mask[ty + th : fy1, :] = cv2.GC_PR_FGD
            mask[: max(ty, 0), :] = cv2.GC_BGD
        mask[yellow > 0] = cv2.GC_BGD
        mask[grass > 0] = cv2.GC_BGD
        cv2.grabCut(small, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    else:
        mask[: int(0.10 * sh), :] = cv2.GC_BGD
        rect = (int(0.03 * sw), int(0.08 * sh), int(0.94 * sw), int(0.84 * sh))
        cv2.grabCut(small, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)

    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)
    if scale != 1.0:
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return _largest_component_mask(binary)


def refine_fish_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Quita pared/hormigón claro, césped superior y cinta de la máscara del pez."""
    binary = (mask > 0.5).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    concrete = (val > 205) & (sat < 40)
    grass = (hue >= 32) & (hue <= 85) & (sat > 55)
    grass[int(0.42 * image.shape[0]) :, :] = False
    yellow = cv2.inRange(hsv, (18, 120, 120), (40, 255, 255)) > 0
    binary[concrete] = 0
    binary[grass] = 0
    binary[yellow] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = _largest_component_mask(binary)
    return cleaned if cleaned is not None else mask


def segment_fish(image: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Detecta el pez: GrabCut guiado por la cinta, con fallback de contornos."""
    tape_bbox = detect_yellow_tape_bbox(image)
    grab = segment_fish_grabcut(image, tape_bbox=tape_bbox)
    if grab is not None:
        grab = refine_fish_mask(image, grab)
        if _mask_looks_like_fish(grab, image):
            logger.info("Pez segmentado con GrabCut (%s).", "cinta" if tape_bbox else "rectángulo")
            return grab, 0.82
    contour = find_fish_by_contour(image)
    if contour is not None:
        logger.info("Pez segmentado con contornos OpenCV.")
        return contour
    return None


# ---------------------------------------------------------------------------
# Escala regla: píxeles por centímetro (Paso D)
# ---------------------------------------------------------------------------


def _ticks_from_profile(profile: np.ndarray, tick_spacing_cm: float) -> float | None:
    """Picos periódicos de un perfil 1D → píxeles por centímetro."""
    profile = profile.astype(np.float64)
    profile = profile - np.median(profile)
    profile[profile < 0] = 0
    if profile.max() <= 0:
        return None

    threshold = float(profile.max()) * 0.40
    min_gap = 6
    peaks: list[int] = []
    i = 1
    n = len(profile)
    while i < n - 1:
        if profile[i] >= threshold and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_gap:
                peaks.append(i)
                i += min_gap
                continue
        i += 1

    if len(peaks) < 5:
        return None

    deltas = np.diff(np.asarray(peaks, dtype=np.float64))
    median = float(np.median(deltas))
    inliers = deltas[(deltas > 0.65 * median) & (deltas < 1.35 * median)]
    if len(inliers) < 4 or tick_spacing_cm <= 0:
        return None

    pixels_per_cm = float(np.median(inliers) / tick_spacing_cm)
    if pixels_per_cm < 2.0 or pixels_per_cm > 500.0:
        return None
    return pixels_per_cm


def _autocorr_lag(profile: np.ndarray, min_lag: int, max_lag: int) -> tuple[int, float] | None:
    """Retardo de máxima autocorrelación (periodo dominante de las marcas)."""
    signal = np.asarray(profile, dtype=np.float64)
    if signal.size < min_lag * 4:
        return None
    signal = signal - np.median(signal)
    if float(signal.std()) < 1e-6:
        return None
    corr = np.correlate(signal, signal, mode="full")
    centered = corr[len(corr) // 2 :]
    max_lag = int(min(max_lag, len(centered) - 1))
    if max_lag <= min_lag:
        return None
    window = centered[: max_lag + 1].copy()
    window[:min_lag] = 0
    lag = int(np.argmax(window))
    energy = float(window[lag])
    if lag < min_lag or energy <= 0:
        return None
    return lag, energy


def _scale_hypotheses(
    period_px: float,
    energy: float,
    band_width_px: float,
    allow_imperial: bool = False,
) -> list[tuple[float, float, str]]:
    """Interpreta un periodo en px como 1 cm / 1 mm / (opcional) pulgadas."""
    hypotheses: list[tuple[float, float, str]] = []
    tick_cm_options: tuple[tuple[float, str], ...] = (
        (1.0, "cm"),
        (0.1, "mm"),
        (5.0, "5cm"),
    )
    if allow_imperial:
        tick_cm_options = tick_cm_options + ((2.54, "inch"), (0.3175, "1/8 inch"))
    for tick_cm, name in tick_cm_options:
        px_per_cm = period_px / tick_cm
        if px_per_cm < 4.0 or px_per_cm > 250.0:
            continue
        visible_cm = band_width_px / px_per_cm
        if visible_cm < 18.0 or visible_cm > 120.0:
            continue
        preference = float(np.exp(-((visible_cm - 50.0) ** 2) / (2.0 * 22.0 ** 2)))
        if name == "cm":
            preference *= 1.45
        hypotheses.append((energy * preference, px_per_cm, name))
    return hypotheses


def estimate_pixels_per_cm_from_ticks(
    image: np.ndarray,
    tick_spacing_cm: float = RULER_TICK_SPACING_CM,
) -> float | None:
    """Estima s a partir de marcas periódicas (cinta amarilla o medidor negro).

    1. Cinta métrica amarilla: autocorrelación en rebanadas de la cinta.
    2. Medidor de pesca: filas del borde superior/inferior (ticks blancos).
    3. Fallback: picos Canny en franjas, asumiendo 1 cm entre marcas.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scored: list[tuple[float, float, str]] = []

    tape = detect_yellow_tape_bbox(image)
    if tape is not None:
        x, y, bw, bh = tape
        band = gray[y : y + max(bh, 2), x : x + max(bw, 2)]
        for frac0, frac1 in ((0.0, 0.35), (0.30, 0.70), (0.65, 1.0)):
            y0 = int(band.shape[0] * frac0)
            y1 = max(y0 + 2, int(band.shape[0] * frac1))
            sl = band[y0:y1, :]
            if sl.size == 0:
                continue
            profile = np.abs(np.diff(sl.astype(np.float64), axis=1, prepend=sl[:, :1])).mean(axis=0)
            result = _autocorr_lag(profile, min_lag=max(8, bw // 250), max_lag=max(40, bw // 8))
            if result is None:
                continue
            lag, energy = result
            scored.extend(_scale_hypotheses(float(lag), energy, float(bw), allow_imperial=True))

    for y0_frac, y1_frac, width_ref in ((0.0, 0.28, float(w)), (0.72, 1.0, float(w))):
        y0, y1 = int(h * y0_frac), max(int(h * y0_frac) + 2, int(h * y1_frac))
        strip = gray[y0:y1, :]
        step = max(1, strip.shape[0] // 40)
        for local_y in range(0, strip.shape[0], step):
            row = strip[local_y].astype(np.float64)
            delta = np.abs(np.diff(row, prepend=row[:1]))
            result = _autocorr_lag(delta, min_lag=8, max_lag=min(80, w // 6))
            if result is None:
                continue
            lag, energy = result
            scored.extend(_scale_hypotheses(float(lag), energy, width_ref, allow_imperial=False))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        _score, px_per_cm, name = scored[0]
        logger.info("Escala por regla (%s): %.3f px/cm.", name, px_per_cm)
        return float(px_per_cm)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    bands = (
        edges[int(h * 0.68) :, :],
        edges[: int(h * 0.32), :],
        edges[:, int(w * 0.68) :],
        edges[:, : int(w * 0.32)],
    )
    for band in bands:
        if band.size == 0:
            continue
        col_energy = band.mean(axis=0)
        row_energy = band.mean(axis=1)
        profile = col_energy if float(col_energy.max()) >= float(row_energy.max()) else row_energy
        scale = _ticks_from_profile(profile, tick_spacing_cm)
        if scale is not None:
            logger.info("Escala por ticks Canny: %.3f px/cm.", scale)
            return scale
    return None


def estimate_pixels_per_cm_from_ruler_bbox(
    ruler_xyxy: Sequence[float],
    ruler_length_cm: float,
) -> float:
    """s = longitud_px_de_la_regla / longitud_real_cm.

    Si YOLO (o un detector futuro) da el bbox de la regla y conocemos su
    largo real, la diagonal o el lado mayor del bbox aproxima d_px_regla.
    """
    x1, y1, x2, y2 = map(float, ruler_xyxy)
    ruler_px = max(abs(x2 - x1), abs(y2 - y1))
    if ruler_length_cm <= 0 or ruler_px <= 0:
        raise ValueError("La regla detectada o su longitud real no son válidas.")
    return ruler_px / ruler_length_cm


# ---------------------------------------------------------------------------
# Visualización debug
# ---------------------------------------------------------------------------


def draw_debug_overlay(
    image: np.ndarray,
    snout: tuple[float, float],
    tail: tuple[float, float],
    length_cm: float,
    mask: np.ndarray | None = None,
    bbox_xyxy: Sequence[float] | None = None,
) -> np.ndarray:
    """Copia de la foto con máscara/caja, línea hocico–cola y texto en cm."""
    canvas = image.copy()
    h, w = canvas.shape[:2]
    stroke = max(2, int(round(min(h, w) / 450)))
    radius = max(6, stroke * 3)
    font_scale = max(0.7, min(h, w) / 550)
    font_thick = max(2, stroke)

    if mask is not None:
        binary = (mask > 0.5).astype(np.uint8)
        if binary.shape[:2] != canvas.shape[:2]:
            binary = cv2.resize(binary, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = np.zeros_like(canvas)
        color[:, :] = (0, 200, 80)
        overlay = np.where(binary[:, :, None] == 1, color, canvas)
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), stroke)

    p1 = (int(round(snout[0])), int(round(snout[1])))
    p2 = (int(round(tail[0])), int(round(tail[1])))
    cv2.line(canvas, p1, p2, (0, 0, 255), stroke + 1, cv2.LINE_AA)
    cv2.circle(canvas, p1, radius, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, p2, radius, (255, 0, 0), -1, cv2.LINE_AA)

    label = f"{length_cm:.1f} cm"
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 12)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    box_tl = (mid[0] - 6, mid[1] - th - 8)
    box_br = (mid[0] + tw + 6, mid[1] + 8)
    cv2.rectangle(canvas, box_tl, box_br, (0, 0, 0), -1)
    cv2.putText(
        canvas,
        label,
        (mid[0], mid[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thick,
        cv2.LINE_AA,
    )
    return canvas


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------


class TroutMeasurer:
    """Mide una trucha sobre regla a partir de una foto local.

    Parameters
    ----------
    model_path:
        Peso YOLO. Cambia a ``models/trout-seg.pt`` cuando tengas un modelo
        entrenado con clases ``trout`` / ``fish``.
    confidence_threshold:
        Umbral ``conf`` de Ultralytics.
    default_pixels_per_cm:
        Escala de laboratorio si no hay parámetro ni regla detectable.
    auto_align:
        Si True, intenta rectificar un tablero rectangular (Paso B).
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        default_pixels_per_cm: float = DEFAULT_PIXELS_PER_CM,
        auto_align: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.default_pixels_per_cm = default_pixels_per_cm
        self.auto_align = auto_align
        logger.info("Cargando modelo YOLO: %s", self.model_path)
        from ultralytics import YOLO  # import diferido: el resto del módulo no exige torch al importar

        self.model = YOLO(self.model_path)
        self.class_names: dict[int, str] = {
            int(k): str(v).lower() for k, v in self.model.names.items()
        }
        self._has_fish_classes = any(name in FISH_CLASS_NAMES for name in self.class_names.values())
        if not self._has_fish_classes:
            logger.warning(
                "El modelo %s no declara clases de pez %s. "
                "Se usará la mayor detección COCO o un contorno OpenCV hasta "
                "que se sustituya por un .pt personalizado.",
                self.model_path,
                sorted(FISH_CLASS_NAMES),
            )

    # -- selección de detección YOLO -------------------------------------------------

    def _fish_class_ids(self) -> list[int]:
        return [idx for idx, name in self.class_names.items() if name in FISH_CLASS_NAMES]

    def _pick_yolo_detection(self, result: Any) -> dict[str, Any] | None:
        """Elige la mejor instancia de pez (o la de mayor confianza/área)."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        # COCO no tiene clase "fish": usar esas cajas (plato, planta, oso...)
        # empeora el MVP. Solo confiamos en YOLO si el .pt declara trucha/pez.
        if not self._has_fish_classes:
            return None

        fish_ids = set(self._fish_class_ids())
        candidates: list[dict[str, Any]] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
            conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            area = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            is_fish = cls_id in fish_ids
            if self._has_fish_classes and not is_fish:
                continue
            candidates.append(
                {
                    "index": i,
                    "cls_id": cls_id,
                    "cls_name": self.class_names.get(cls_id, "unknown"),
                    "confidence": conf,
                    "xyxy": xyxy,
                    "area": area,
                    "is_fish": is_fish,
                }
            )

        if not candidates:
            return None
        # Prioriza clase pez; desempata por confianza * área.
        candidates.sort(key=lambda c: (c["is_fish"], c["confidence"] * (1.0 + np.log1p(c["area"]))), reverse=True)
        return candidates[0]

    def _extract_extrema(self, result: Any, detection: dict[str, Any]) -> tuple[
        tuple[float, float],
        tuple[float, float],
        np.ndarray | None,
    ]:
        """Hocico/cola desde máscara, keypoints o bbox (en ese orden)."""
        idx = detection["index"]
        mask_np: np.ndarray | None = None
        orig_h, orig_w = result.orig_shape[:2]

        masks = getattr(result, "masks", None)
        if masks is not None:
            # Polígono en coordenadas de la imagen original: más fiable que
            # reescalar el tensor interno (suele ser 640×640).
            xy_list = getattr(masks, "xy", None)
            if xy_list is not None and len(xy_list) > idx and xy_list[idx] is not None:
                poly = np.asarray(xy_list[idx], dtype=np.float64)
                if len(poly) >= 2:
                    snout, tail = extrema_along_principal_axis(poly)
                    if getattr(masks, "data", None) is not None and len(masks.data) > idx:
                        raw = masks.data[idx].cpu().numpy()
                        mask_np = cv2.resize(
                            raw.astype(np.float32),
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    return snout, tail, mask_np

            if getattr(masks, "data", None) is not None and len(masks.data) > idx:
                mask_np = masks.data[idx].cpu().numpy()
                if mask_np.shape != (orig_h, orig_w):
                    mask_np = cv2.resize(
                        mask_np.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                try:
                    snout, tail = extrema_from_mask(mask_np)
                    return snout, tail, mask_np
                except FishNotDetectedError:
                    logger.debug("Máscara inválida; se prueba keypoints/bbox.")

        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) > idx:
            kxy = keypoints.xy[idx].cpu().numpy()
            try:
                snout, tail = extrema_from_keypoints(kxy)
                return snout, tail, mask_np
            except FishNotDetectedError:
                logger.debug("Keypoints insuficientes; se usa bbox.")

        snout, tail = extrema_from_bbox(detection["xyxy"])
        return snout, tail, mask_np

    def _resolve_scale(
        self,
        image: np.ndarray,
        pixels_per_cm: float | None,
        ruler_xyxy: Sequence[float] | None = None,
        ruler_length_cm: float | None = DEFAULT_RULER_LENGTH_CM,
    ) -> tuple[float, str]:
        """Devuelve (s, origen) con s = pixels_per_cm usado.

        Prioridad:
        1. Parámetro explícito (calibración manual / API).
        2. Bbox de regla + longitud real conocida.
        3. Ticks periódicos detectados en la imagen.
        4. DEFAULT_PIXELS_PER_CM (configurable en config.py / env).
        """
        if pixels_per_cm is not None:
            if pixels_per_cm <= 0:
                raise ValueError("pixels_per_cm debe ser un número positivo.")
            return float(pixels_per_cm), "parameter"

        if ruler_xyxy is not None and ruler_length_cm:
            scale = estimate_pixels_per_cm_from_ruler_bbox(ruler_xyxy, ruler_length_cm)
            return scale, "ruler_bbox"

        tick_scale = estimate_pixels_per_cm_from_ticks(image)
        if tick_scale is not None:
            return tick_scale, "ruler_ticks"

        logger.warning(
            "No se detectó regla; se usa DEFAULT_PIXELS_PER_CM=%.3f. "
            "Calibra con --px-per-cm o pixels_per_cm en la API.",
            self.default_pixels_per_cm,
        )
        return float(self.default_pixels_per_cm), "default"

    # -- API pública -----------------------------------------------------------------

    def measure(
        self,
        image_path: str | Path,
        pixels_per_cm: float | None = None,
        board_corners: Sequence[Sequence[float]] | None = None,
        output_dir: str | Path = OUTPUT_DIR,
        ruler_length_cm: float | None = DEFAULT_RULER_LENGTH_CM,
    ) -> MeasurementResult:
        """Mide el pez de ``image_path`` y guarda un overlay en ``output/``.

        Raises
        ------
        ImageNotFoundError, InvalidImageError, FishNotDetectedError, ValueError
        """
        image_path = Path(image_path)
        image = read_image(image_path)

        if board_corners is not None:
            image, _ = warp_with_homography(image, board_corners)
        elif self.auto_align:
            image = crop_interest_region(image)

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not results:
            raise FishNotDetectedError("YOLO no devolvió resultados.")

        result = results[0]
        detection = self._pick_yolo_detection(result)

        mask: np.ndarray | None = None
        bbox: Sequence[float] | None = None
        source = "yolo"

        if detection is not None:
            snout, tail, mask = self._extract_extrema(result, detection)
            bbox = detection["xyxy"]
            confidence = float(detection["confidence"])
            source = "yolo"
        else:
            fallback = segment_fish(image)
            if fallback is None:
                raise FishNotDetectedError(
                    "No se detectó ningún pez. Prueba un modelo .pt entrenado "
                    f"en truchas (ahora mismo: {self.model_path}) o una foto "
                    "más nítida del pez sobre la regla."
                )
            mask, confidence = fallback
            snout, tail = extrema_from_mask(mask)
            source = "opencv_contour"
            logger.info("YOLO no encontró pez; se usó segmentación OpenCV.")

        # Distancia euclidiana en el plano imagen (píxeles).
        d_px = euclidean_distance(snout, tail)
        if d_px < 1.0:
            raise FishNotDetectedError("Los extremos detectados están demasiado juntos.")

        scale, scale_source = self._resolve_scale(
            image,
            pixels_per_cm=pixels_per_cm,
            ruler_length_cm=ruler_length_cm,
        )
        # longitud_cm = d_px / pixels_per_cm
        length_cm = d_px / scale

        overlay = draw_debug_overlay(
            image,
            snout=snout,
            tail=tail,
            length_cm=length_cm,
            mask=mask,
            bbox_xyxy=bbox,
        )
        out_path = Path(output_dir) / f"res_{image_path.stem}.jpg"
        write_image(out_path, overlay)

        return MeasurementResult(
            success=True,
            length_cm=length_cm,
            confidence=confidence,
            pixels_measured=d_px,
            pixels_per_cm_used=scale,
            processed_image_path=str(out_path),
            snout_xy=snout,
            tail_xy=tail,
            detection_source=source,
            scale_source=scale_source,
            extra={
                "class_name": None if detection is None else detection["cls_name"],
                "model_path": self.model_path,
            },
        )

    def measure_array(
        self,
        image: np.ndarray,
        source_name: str = "upload",
        pixels_per_cm: float | None = None,
        output_dir: str | Path = OUTPUT_DIR,
    ) -> MeasurementResult:
        """Igual que ``measure`` pero desde un array BGR (endpoint de upload)."""
        tmp_dir = Path(output_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"_src_{source_name}.jpg"
        write_image(tmp_path, image)
        try:
            return self.measure(tmp_path, pixels_per_cm=pixels_per_cm, output_dir=output_dir)
        finally:
            if tmp_path.exists() and tmp_path.name.startswith("_src_"):
                tmp_path.unlink(missing_ok=True)
