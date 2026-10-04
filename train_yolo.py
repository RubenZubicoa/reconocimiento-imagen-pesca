"""Entrena un YOLO-Seg de truchas y deja el mejor peso en models/trout-seg.pt.

Uso (con el venv activado)::

    python train_yolo.py --epochs 80 --imgsz 640

Requisitos previos
------------------
1. Fotos de trucha sobre regla (las de assets/ valen como inicio).
2. Polígonos etiquetados en formato YOLO-Seg, en::

       datasets/trout/images/train  +  labels/train
       datasets/trout/images/val    +  labels/val

   Cada .txt comparte nombre con su imagen. Una línea por objeto::

       0 x1 y1 x2 y2 ... xn yn

   Coordenadas normalizadas 0–1. La clase ``0`` es ``trout`` (ver data.yaml).

Herramientas de etiquetado: Roboflow, CVAT o Label Studio (export YOLO-Seg).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from config import MODELS_DIR, PROJECT_ROOT

DATA_YAML = PROJECT_ROOT / "datasets" / "trout" / "data.yaml"
DEFAULT_BASE_MODEL = "yolov8n-seg.pt"


def _count_labeled_pairs(split: str) -> int:
    labels = PROJECT_ROOT / "datasets" / "trout" / "labels" / split
    if not labels.is_dir():
        return 0
    return sum(1 for path in labels.glob("*.txt") if path.stat().st_size > 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena YOLO-Seg para truchas.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Ruta al data.yaml.")
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Peso de partida (transfer learning). yolov8s-seg.pt es más preciso.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="trout-seg")
    parser.add_argument(
        "--device",
        default="",
        help="Vacío = auto. Usa 0 para la primera GPU, cpu si no hay CUDA.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    n_train = _count_labeled_pairs("train")
    n_val = _count_labeled_pairs("val")
    if n_train < 10:
        raise SystemExit(
            f"Hacen falta etiquetas en datasets/trout/labels/train (hay {n_train}). "
            "Etiqueta al menos ~30–50 fotos (80/20 train/val) antes de entrenar."
        )
    if n_val < 1:
        raise SystemExit("Añade al menos unas fotos en datasets/trout/labels/val.")

    print(f"Dataset: {n_train} train / {n_val} val")
    model = YOLO(args.base_model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=str(PROJECT_ROOT / "runs"),
        device=args.device or None,
        # Augmentaciones útiles para fotos de campo (luz, ángulo, regla).
        degrees=15,
        fliplr=0.5,
        mosaic=0.8,
        close_mosaic=10,
        patience=20,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"No se encontró {best}. Revisa la carpeta runs/.")

    dest = MODELS_DIR / "trout-seg.pt"
    shutil.copy2(best, dest)
    print(f"\nModelo copiado a {dest}")
    print("Prueba:")
    print(f"  python cli.py --image assets/1.jpg --model {dest}")
    print("O en PowerShell:")
    print(f"  $env:TROUT_MODEL_PATH = '{dest}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
