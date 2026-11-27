"""Skrip Pelatihan Model Random Forest untuk Sistem Deteksi Stres (StressGuard).

Skrip ini menjalankan seluruh alur pelatihan secara otomatis:
1. Membaca dan membersihkan dataset student_lifestyle_dataset.csv.
2. Memilih 5 fitur perilaku harian (membuang Student_ID dan GPA).
3. Membagi data latih (80%) dan data uji (20%) secara terstratifikasi.
4. Melatih model RandomForestClassifier (max_depth=6, n_estimators=100).
5. Mengevaluasi performa dan menyimpan grafik evaluasi_model.png.
6. Mengekspor model terlatih ke prediction/ml_models/model_stres.joblib.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. KONFIGURASI & KONTRAK FITUR (HARUS IDENTIK DENGAN views.py)
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "student_lifestyle_dataset.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "prediction" / "ml_models" / "model_stres.joblib"
EVAL_CHART_PATH = BASE_DIR / "evaluasi_model.png"

FEATURE_COLUMNS = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
]
FEATURE_LABELS_ID = [
    "Belajar / Kerja Fokus",
    "Ekstrakurikuler",
    "Tidur / Istirahat",
    "Interaksi Sosial",
    "Aktivitas Fisik",
]
TARGET_COLUMN = "Stress_Level"
CLASS_ORDER = ["Low", "Moderate", "High"]


def load_and_clean_dataset(csv_path: Path) -> pd.DataFrame:
    """Membaca file CSV, merapikan nama kolom, dan memeriksa kebersihan data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    missing_total = int(df.isnull().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    print(f"[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[INFO] Nilai kosong (missing values): {missing_total}")
    print(f"[INFO] Baris duplikat: {duplicate_total}")

    if missing_total > 0:
        df = df.dropna()
        print(f"[INFO] Baris setelah pembersihan missing values: {df.shape[0]}")

    return df


def save_evaluation_chart(
    y_test: pd.Series,
    y_pred: pd.Series,
    feature_importances: list[float],
    output_path: Path,
) -> None:
    """Membuat dan menyimpan gambar Confusion Matrix & Feature Importance."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    fig.patch.set_facecolor("#F6F5F2")

    # Panel Kiri: Confusion Matrix (3x3)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        linewidths=1,
        linecolor="#F6F5F2",
        annot_kws={"size": 12, "weight": "bold"},
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Data Uji 20%)", fontsize=12, fontweight="bold", pad=12, color="#141413")
    axes[0].set_xlabel("Prediksi Model", fontsize=10, labelpad=8, color="#141413")
    axes[0].set_ylabel("Label Sebenarnya", fontsize=10, labelpad=8, color="#141413")

    # Panel Kanan: Feature Importance
    importance_df = pd.DataFrame(
        {"Fitur": FEATURE_LABELS_ID, "Bobot": feature_importances}
    ).sort_values("Bobot", ascending=True)

    bars = axes[1].barh(
        importance_df["Fitur"],
        importance_df["Bobot"],
        color="#1F2937",
        height=0.55,
    )
    axes[1].set_facecolor("#FFFFFF")
    axes[1].set_title("Tingkat Pengaruh Fitur (Feature Importance)", fontsize=12, fontweight="bold", pad=12, color="#141413")
    axes[1].set_xlabel("Skor Kepentingan Relatif", fontsize=10, labelpad=8, color="#141413")
    axes[1].set_xlim(0, max(importance_df["Bobot"]) * 1.18)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4, color="#A8A69E")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    for bar in bars:
        width = bar.get_width()
        axes[1].text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=9.5,
            color="#141413",
        )

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Grafik evaluasi disimpan ke: {output_path}")


def train_and_export() -> None:
    """Menjalankan pipeline training, evaluasi metrik, dan ekspor model."""
    df = load_and_clean_dataset(DATASET_PATH)

    # Seleksi fitur: membuang Student_ID dan GPA sesuai PRD Bagian 6.2
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"[INFO] Split data -> Latih: {len(X_train)} baris | Uji: {len(X_test)} baris")

    # Pelatihan Random Forest dengan pembatasan max_depth=6 agar tidak overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluasi pada data latih dan data uji
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    overfitting_gap = abs(train_acc - test_acc)

    print("\n=== HASIL EVALUASI MODEL RANDOM FOREST ===")
    print(f"Akurasi Data Latih : {train_acc:.2f}%")
    print(f"Akurasi Data Uji   : {test_acc:.2f}%")
    print(f"Selisih (Gap)      : {overfitting_gap:.2f}%")
    print("\nClassification Report (Data Uji):")
    print(classification_report(y_test, y_test_pred, labels=CLASS_ORDER, digits=4))

    print("Feature Importances:")
    for col, imp in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"  - {col:<32}: {imp:.4f}")

    # Simpan grafik evaluasi
    save_evaluation_chart(
        y_test=y_test,
        y_pred=y_test_pred,
        feature_importances=list(model.feature_importances_),
        output_path=EVAL_CHART_PATH,
    )

    # Simpan model terlatih (.joblib)
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"[INFO] Model berhasil diekspor ke: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_and_export()
