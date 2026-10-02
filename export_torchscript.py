"""
export_torchscript.py
=====================
Script untuk mengekspor model YOLO best.pt ke format TorchScript (.pt)
yang bisa di-load oleh PyTorch Mobile (pytorch_android).

Cara pakai:
    pip install ultralytics torch
    python export_torchscript.py

Output: best_torchscript.pt — taruh di DiadetApp/app/src/main/assets/ dengan nama best.pt
"""

from pathlib import Path
from ultralytics import YOLO
import torch

MODEL_PATH = "Andidet.AI/Diadet AI/best.pt"
OUTPUT_DIR = "DiadetApp/app/src/main/assets"
OUTPUT_NAME = "best.pt"

def export():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Export ke TorchScript format
    # format='torchscript' → kompatibel dengan pytorch_android Module.load()
    print("Exporting to TorchScript...")
    exported = model.export(
        format="torchscript",
        imgsz=640,
        optimize=False,   # False = lebih kompatibel dengan mobile
        half=False,        # False = float32, lebih stabil di Android
    )

    # Salin hasil ke assets
    src = Path(exported)
    dst = Path(OUTPUT_DIR) / OUTPUT_NAME
    dst.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy2(src, dst)
    print(f"✅ Model exported to: {dst}")
    print(f"   File size: {dst.stat().st_size / 1024 / 1024:.1f} MB")

    # Verifikasi bisa di-load
    print("Verifying TorchScript load...")
    scripted = torch.jit.load(str(dst))
    print(f"✅ Verification OK: {type(scripted)}")

if __name__ == "__main__":
    export()
