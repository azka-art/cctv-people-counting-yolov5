"""
Standalone visualization helper for People Detection.

Usage:
    python -m src.inference.visualize \
        --input data/mot20/train/MOT20-01/img1/000142.jpg \
        --output assets/sample_outputs/case_01_occlusion.jpg \
        --conf 0.4
"""

import argparse
import sys
from pathlib import Path

from src.inference.inference_image import run_image_inference


def main():
    parser = argparse.ArgumentParser(description="People Detection - Visualize Single Image")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to output annotated image")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    detections, count = run_image_inference(args.input, args.output, args.conf, args.device)
    print(f"[RESULT] {count} people detected")
    for i, det in enumerate(detections):
        print(f"  [{i+1}] bbox=({det['x1']},{det['y1']})-({det['x2']},{det['y2']}) score={det['score']}")


if __name__ == "__main__":
    main()
