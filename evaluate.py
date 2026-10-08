"""
evaluate.py
===========
Script untuk mengevaluasi performa model YOLO Diabetes Detection pada dataset pengujian (Test Set).
Dataset default: Testing/preprocessedcropped/test

Output:
- evaluation_results/confusion_matrix.png
- evaluation_results/confidence_distribution.png
- evaluation_results/predictions_test.csv
- evaluation_results/evaluation_metrics.json
"""

import os
import glob
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from ultralytics import YOLO

# Path konfigurasi
MODEL_PATH = "Andidet.AI/Diadet AI/best.pt"
DATASET_PATH = "Testing/preprocessedcropped/test"
OUTPUT_DIR = "evaluation_results"

def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Memuat model YOLO dari:", MODEL_PATH)
    print("Dataset pengujian:", DATASET_PATH)
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {DATASET_PATH}")
        
    model = YOLO(MODEL_PATH)
    
    # Kategori kelas
    target_classes = ["diabetes", "nondiabetes"]
    
    records = []
    latencies = []
    
    for true_cls in target_classes:
        folder = os.path.join(DATASET_PATH, true_cls)
        image_files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
        print(f"Memproses kelas '{true_cls}': {len(image_files)} gambar ditemukan.")
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            start_t = time.perf_counter()
            results = model(img_path, verbose=False)[0]
            lat = (time.perf_counter() - start_t) * 1000  # ms
            latencies.append(lat)
            
            boxes = results.boxes
            if len(boxes) > 0:
                cls_id = int(boxes.cls[0].item())
                pred_label_raw = model.names[cls_id]  # 'Diabetes' atau 'Nondiabetes'
                conf = float(boxes.conf[0].item())
                num_boxes = len(boxes)
                box_coords = [round(float(c), 2) for c in boxes.xyxy[0].tolist()]
            else:
                pred_label_raw = "None"
                conf = 0.0
                num_boxes = 0
                box_coords = []
                
            pred_cls = pred_label_raw.lower()
            is_correct = (pred_cls == true_cls)
            
            # Probability score for ROC AUC (probabilitas kelas 'diabetes')
            # Jika terdeteksi Diabetes: score = conf
            # Jika terdeteksi Nondiabetes: score = 1.0 - conf
            prob_diabetes = conf if pred_cls == "diabetes" else (1.0 - conf if pred_cls == "nondiabetes" else 0.5)
            
            records.append({
                "filename": filename,
                "filepath": img_path,
                "true_label": true_cls,
                "predicted_label": pred_cls,
                "confidence": conf,
                "prob_diabetes": prob_diabetes,
                "num_detections": num_boxes,
                "bounding_box_xyxy": str(box_coords),
                "is_correct": is_correct,
                "latency_ms": round(lat, 2)
            })
            
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "predictions_test.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Data prediksi detail disimpan di: {csv_path}")
    
    # Metrik Evaluasi
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()
    
    labels = ["diabetes", "nondiabetes"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # cm layout:
    # [[TP, FN],
    #  [FP, TN]] (dengan diabetes sebagai positif)
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    
    acc = accuracy_score(y_true, y_pred)
    prec_diabetes = precision_score(y_true, y_pred, pos_label="diabetes")
    rec_diabetes = recall_score(y_true, y_pred, pos_label="diabetes")  # sensitivity
    f1_diab = f1_score(y_true, y_pred, pos_label="diabetes")
    
    prec_nondiab = precision_score(y_true, y_pred, pos_label="nondiabetes")
    rec_nondiab = recall_score(y_true, y_pred, pos_label="nondiabetes")  # specificity
    f1_nondiab = f1_score(y_true, y_pred, pos_label="nondiabetes")
    
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    
    # AUC score
    y_true_binary = [1 if t == "diabetes" else 0 for t in y_true]
    y_prob = df["prob_diabetes"].tolist()
    auc_score = roc_auc_score(y_true_binary, y_prob)
    
    clf_report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    
    metrics = {
        "dataset_path": DATASET_PATH,
        "total_images": len(df),
        "total_diabetes": sum(1 for t in y_true if t == "diabetes"),
        "total_nondiabetes": sum(1 for t in y_true if t == "nondiabetes"),
        "correct_predictions": int(df["is_correct"].sum()),
        "accuracy": float(acc),
        "confusion_matrix": {
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn)
        },
        "diabetes_metrics": {
            "precision": float(prec_diabetes),
            "recall_sensitivity": float(rec_diabetes),
            "f1_score": float(f1_diab),
            "support": int(tp + fn)
        },
        "nondiabetes_metrics": {
            "precision": float(prec_nondiab),
            "recall_specificity": float(rec_nondiab),
            "f1_score": float(f1_nondiab),
            "support": int(tn + fp)
        },
        "macro_avg": {
            "f1_score": float(f1_macro)
        },
        "weighted_avg": {
            "f1_score": float(f1_weighted)
        },
        "roc_auc": float(auc_score),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies))
        }
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Metrik evaluasi disimpan di: {metrics_path}")
    
    # 1. Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Diabetes", "Nondiabetes"],
        yticklabels=["Diabetes", "Nondiabetes"],
        title=f"Confusion Matrix (Accuracy: {acc*100:.1f}%)",
        ylabel="Ground Truth (Label Asli)",
        xlabel="Predicted Label (Prediksi)"
    )
    
    # Tambahkan angka di setiap sel
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, weight="bold"
            )
            
    fig.tight_layout()
    cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_plot_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Gambar Confusion Matrix disimpan di: {cm_plot_path}")
    
    # 2. Plot ROC Curve & Confidence Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("Receiver Operating Characteristic (ROC)")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Confidence distribution by correctness
    correct_confs = df[df["is_correct"]]["confidence"]
    wrong_confs = df[~df["is_correct"]]["confidence"]
    
    ax2.hist(correct_confs, bins=10, alpha=0.7, color="green", label=f"Benar (n={len(correct_confs)})")
    if len(wrong_confs) > 0:
        ax2.hist(wrong_confs, bins=10, alpha=0.7, color="red", label=f"Salah (n={len(wrong_confs)})")
    ax2.set_xlabel("Tingkat Keyakinan (Confidence Score)")
    ax2.set_ylabel("Jumlah Gambar")
    ax2.set_title("Distribusi Confidence Prediksi")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    roc_plot_path = os.path.join(OUTPUT_DIR, "roc_and_confidence.png")
    fig.savefig(roc_plot_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Gambar ROC & Distribusi disimpan di: {roc_plot_path}")
    
    # Cetak Rangkuman di Terminal
    print("\n" + "=" * 60)
    print("HASIL EVALUASI MODEL:")
    print("=" * 60)
    print(f"Total Gambar Diuji   : {len(df)}")
    print(f"Prediksi Benar       : {int(df['is_correct'].sum())} / {len(df)}")
    print(f"Akurasi Keseluruhan  : {acc * 100:.2f}%")
    print(f"ROC-AUC Score        : {auc_score:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  True Positive (Diabetes -> Diabetes)       : {tp}")
    print(f"  False Negative (Diabetes -> Nondiabetes)   : {fn}")
    print(f"  False Positive (Nondiabetes -> Diabetes)   : {fp}")
    print(f"  True Negative (Nondiabetes -> Nondiabetes) : {tn}")
    print("-" * 60)
    print("Detail Per Kelas:")
    print(f"  [Diabetes]")
    print(f"    - Precision    : {prec_diabetes * 100:.2f}%")
    print(f"    - Recall (Sens): {rec_diabetes * 100:.2f}%")
    print(f"    - F1-Score     : {f1_diab * 100:.2f}%")
    print(f"  [Nondiabetes]")
    print(f"    - Precision    : {prec_nondiab * 100:.2f}%")
    print(f"    - Recall (Spec): {rec_nondiab * 100:.2f}%")
    print(f"    - F1-Score     : {f1_nondiab * 100:.2f}%")
    print("-" * 60)
    print(f"Inference Latency: Rata-rata {np.mean(latencies):.2f} ms / gambar")
    
    if len(wrong_confs) > 0:
        print("-" * 60)
        print("Daftar Gambar yang Salah Diprediksi (Misclassified):")
        for idx, row in df[~df["is_correct"]].iterrows():
            print(f"  - File: {row['filename']} | Asli: {row['true_label']} | Prediksi: {row['predicted_label']} | Conf: {row['confidence']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
