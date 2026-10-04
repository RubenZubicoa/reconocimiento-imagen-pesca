"""CLI de prueba del medidor de truchas.

Ejemplos::

    python cli.py --image assets/sample_trout_ruler.png --px-per-cm 12.5
    python cli.py --image foto.jpg --model models/trout-seg.pt --auto-align
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import DEFAULT_MODEL_PATH, DEFAULT_PIXELS_PER_CM, OUTPUT_DIR
from exceptions import TroutMeasurerError
from trout_measurer import TroutMeasurer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estima la longitud (cm) de una trucha sobre una regla de pesca.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Ruta local de la foto (JPG/PNG).",
    )
    parser.add_argument(
        "--px-per-cm",
        type=float,
        default=None,
        dest="pixels_per_cm",
        help=(
            "Escala de calibración en píxeles por centímetro. "
            f"Si se omite, se intenta leer la regla o se usa {DEFAULT_PIXELS_PER_CM}."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Ruta al peso YOLO (.pt). Por defecto yolov8n-seg.pt (se descarga solo).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Carpeta donde guardar el overlay de depuración.",
    )
    parser.add_argument(
        "--auto-align",
        action="store_true",
        help="Intenta rectificar el tablero (homografía) antes de medir.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Imprime el resultado completo en JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    image_path = Path(args.image)
    try:
        measurer = TroutMeasurer(
            model_path=args.model,
            auto_align=args.auto_align,
        )
        result = measurer.measure(
            image_path=image_path,
            pixels_per_cm=args.pixels_per_cm,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as exc:
        print(f"Error: no se encontró el archivo: {exc}", file=sys.stderr)
        return 2
    except TroutMeasurerError as exc:
        print(f"Error de medición: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Parámetro inválido: {exc}", file=sys.stderr)
        return 2

    payload = result.to_api_dict()
    payload["detection_source"] = result.detection_source
    payload["scale_source"] = result.scale_source

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Longitud estimada : {payload['length_cm']:.2f} cm")
        print(f"Confianza         : {payload['confidence']:.2f}")
        print(f"Píxeles medidos   : {payload['pixels_measured']:.2f}")
        print(f"px/cm usados      : {payload['pixels_per_cm_used']:.4f} ({result.scale_source})")
        print(f"Overlay           : {payload['processed_image_path']}")
        print(f"Fuente detección  : {result.detection_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
