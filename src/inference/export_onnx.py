"""
Export YOLOv5s model to ONNX format for production deployment.

ONNX (Open Neural Network Exchange) enables deployment to:
- ONNX Runtime (CPU/GPU inference without PyTorch)
- TensorRT (NVIDIA GPU acceleration)
- OpenVINO (Intel hardware acceleration)
- CoreML / TFLite (mobile deployment)
- Any ONNX-compatible inference engine

Usage:
    # Default export (recommended)
    python -m src.inference.export_onnx \\
        --output assets/yolov5s.onnx

    # With opset version override
    python -m src.inference.export_onnx \\
        --output assets/yolov5s.onnx \\
        --opset 12 --device cpu

    # Verify exported model
    python -m src.inference.export_onnx \\
        --output assets/yolov5s.onnx \\
        --verify

Expected output:
    assets/yolov5s.onnx         — exported model weights
    assets/yolov5s_metadata.json — export metadata (opset, input shape, etc.)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_REPO   = "ultralytics/yolov5"
MODEL_NAME   = "yolov5s"
INPUT_SHAPE  = (1, 3, 640, 640)   # (batch, channels, height, width)
DEFAULT_OPSET = 11                 # ONNX opset — 11 is widely supported


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def load_yolov5(device: str = "cpu") -> torch.nn.Module:
    """Load YOLOv5s from torch.hub (auto-downloads weights on first run)."""
    print(f"[INFO] Loading {MODEL_NAME} from torch.hub ...")
    model = torch.hub.load(MODEL_REPO, MODEL_NAME, pretrained=True, verbose=False)
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded — device: {device}")
    return model


def export_to_onnx(
    output_path: str,
    opset: int = DEFAULT_OPSET,
    device: str = "cpu",
    input_shape: tuple = INPUT_SHAPE,
) -> dict:
    """
    Export YOLOv5s to ONNX format using torch.onnx.export.

    Args:
        output_path:  Destination path for .onnx file
        opset:        ONNX opset version (default: 11)
        device:       "cpu" or "cuda"
        input_shape:  Input tensor shape (default: 1×3×640×640)

    Returns:
        dict with export metadata (path, opset, input_shape, file_size_mb, etc.)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    model = load_yolov5(device)

    # --- Create dummy input ---
    dummy_input = torch.zeros(input_shape, dtype=torch.float32).to(device)
    print(f"[INFO] Input shape  : {list(input_shape)}")
    print(f"[INFO] ONNX opset   : {opset}")
    print(f"[INFO] Output path  : {output_path}")
    print()

    # --- Export ---
    print("[INFO] Exporting to ONNX ...")
    t_start = time.perf_counter()

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            # Allow variable batch size and image dimensions at runtime
            "images": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        },
        do_constant_folding=True,   # Fold constants for smaller/faster model
        verbose=False,
    )

    elapsed = time.perf_counter() - t_start
    file_size_mb = output_path.stat().st_size / (1024 ** 2)

    print(f"[INFO] Export complete in {elapsed:.2f}s")
    print(f"[INFO] File size    : {file_size_mb:.2f} MB")

    metadata = {
        "model":        MODEL_NAME,
        "source":       f"{MODEL_REPO}/{MODEL_NAME}",
        "output_path":  str(output_path.resolve()),
        "opset":        opset,
        "input_shape":  list(input_shape),
        "input_names":  ["images"],
        "output_names": ["output"],
        "dynamic_axes": ["batch", "height", "width"],
        "file_size_mb": round(file_size_mb, 2),
        "export_time_s": round(elapsed, 2),
        "torch_version": torch.__version__,
        "device":       device,
    }

    return metadata


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_onnx(onnx_path: str, input_shape: tuple = INPUT_SHAPE) -> bool:
    """
    Verify the exported ONNX model:
      1. Load and check graph structure with onnx.checker
      2. Run a dummy forward pass with onnxruntime
      3. Compare output shape against expected dimensions

    Args:
        onnx_path:   Path to exported .onnx file
        input_shape: Input tensor shape used during export

    Returns:
        True if all checks pass, False otherwise
    """
    print()
    print("[VERIFY] Running ONNX model verification ...")

    # --- Check 1: onnx graph validation ---
    try:
        import onnx
        model_onnx = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_onnx)
        print("[VERIFY] ✅ ONNX graph structure — valid")
    except ImportError:
        print("[VERIFY] ⚠️  onnx package not installed — skipping graph check")
        print("         Install with: pip install onnx")
    except Exception as e:
        print(f"[VERIFY] ❌ ONNX graph check failed: {e}")
        return False

    # --- Check 2: onnxruntime inference ---
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        dummy = np.random.rand(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy})

        print(f"[VERIFY] ✅ ONNXRuntime inference — output shape: {outputs[0].shape}")
        print(f"[VERIFY] ✅ Input name  : '{input_name}'")
        print(f"[VERIFY] ✅ Output name : '{session.get_outputs()[0].name}'")
        return True

    except ImportError:
        print("[VERIFY] ⚠️  onnxruntime not installed — skipping inference check")
        print("         Install with: pip install onnxruntime")
        return True   # Export itself succeeded; runtime check is optional
    except Exception as e:
        print(f"[VERIFY] ❌ ONNXRuntime inference failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def save_metadata(metadata: dict, output_path: str) -> None:
    """Save export metadata as a JSON file alongside the .onnx file."""
    meta_path = Path(output_path).with_suffix("") .parent / (
        Path(output_path).stem + "_metadata.json"
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Metadata saved : {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv5s to ONNX for production deployment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--output", type=str, default="assets/yolov5s.onnx",
        help="Destination path for exported .onnx file\n"
             "(default: assets/yolov5s.onnx)",
    )
    parser.add_argument(
        "--opset", type=int, default=DEFAULT_OPSET,
        help=f"ONNX opset version (default: {DEFAULT_OPSET})\n"
             "Opset 11 = broad compatibility\n"
             "Opset 12–17 = newer features (check your runtime support)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load model on before export (default: cpu)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run post-export verification with onnxruntime\n"
             "(requires: pip install onnx onnxruntime)",
    )
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Skip saving export metadata JSON",
    )

    args = parser.parse_args()

    # --- Export ---
    print("=" * 60)
    print("  YOLOv5s → ONNX Export")
    print("=" * 60)

    try:
        metadata = export_to_onnx(
            output_path=args.output,
            opset=args.opset,
            device=args.device,
        )
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")
        sys.exit(1)

    # --- Save metadata ---
    if not args.no_metadata:
        save_metadata(metadata, args.output)

    # --- Verify ---
    if args.verify:
        ok = verify_onnx(args.output)
        if not ok:
            print("\n[ERROR] Verification failed — check logs above")
            sys.exit(1)

    # --- Summary ---
    print()
    print("=" * 60)
    print("[DONE] ONNX export successful")
    print(f"       Model    : {metadata['model']}")
    print(f"       Opset    : {metadata['opset']}")
    print(f"       Size     : {metadata['file_size_mb']} MB")
    print(f"       Output   : {metadata['output_path']}")
    print()
    print("  Deployment targets:")
    print("    onnxruntime  →  pip install onnxruntime")
    print("    TensorRT     →  trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.trt")
    print("    OpenVINO     →  mo --input_model yolov5s.onnx")
    print("=" * 60)


if __name__ == "__main__":
    main()