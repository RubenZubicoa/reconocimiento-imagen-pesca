"""Genera una foto sintética de trucha + regla para probar el pipeline sin dataset."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def generate_sample(path: Path | None = None, pixels_per_cm: float = 12.5) -> Path:
    """Dibuja un pez de 34.5 cm sobre una regla con ticks cada 1 cm.

    Con px/cm = 12.5 → longitud del pez en píxeles = 34.5 * 12.5 = 431.25 px.
    """
    path = path or (ASSETS_DIR / "sample_trout_ruler.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = 420, 900
    image = np.full((height, width, 3), 230, dtype=np.uint8)

    # Tablero tipo medidor (beige).
    cv2.rectangle(image, (40, 80), (860, 340), (210, 225, 235), -1)

    # Regla inferior: ticks cada 1 cm = pixels_per_cm píxeles.
    ruler_y0, ruler_y1 = 300, 340
    ruler_x0 = 80
    cv2.rectangle(image, (ruler_x0, ruler_y0), (820, ruler_y1), (245, 245, 245), -1)
    n_cm = 56
    for cm in range(n_cm + 1):
        x = int(round(ruler_x0 + cm * pixels_per_cm))
        if x > 815:
            break
        tick_h = 28 if cm % 5 == 0 else 16
        cv2.line(image, (x, ruler_y1), (x, ruler_y1 - tick_h), (20, 20, 20), 1)
        if cm % 5 == 0:
            cv2.putText(
                image,
                str(cm),
                (x - 8, ruler_y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

    # Trucha: elipse alargada de 34.5 cm de eje mayor.
    length_cm = 34.5
    fish_len_px = int(round(length_cm * pixels_per_cm))
    fish_h_px = 48
    cx, cy = 80 + fish_len_px // 2 + 20, 200
    axes = (fish_len_px // 2, fish_h_px // 2)
    cv2.ellipse(image, (cx, cy), axes, 0, 0, 360, (20, 140, 55), -1)
    # Aleta caudal (cola) y hocico para que el contorno sea más "de pez".
    tail_x = cx - axes[0]
    snout_x = cx + axes[0]
    pts = np.array(
        [[tail_x - 18, cy], [tail_x + 8, cy - 22], [tail_x + 8, cy + 22]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, pts, (30, 70, 40))
    cv2.circle(image, (snout_x - 10, cy - 6), 4, (15, 15, 15), -1)

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen de prueba.")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
    return path


if __name__ == "__main__":
    out = generate_sample()
    print(f"Imagen de prueba escrita en {out}")
