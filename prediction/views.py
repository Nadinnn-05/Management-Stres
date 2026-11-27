"""Logika utama backend StressGuard: validasi input, prediksi ML, dan rekomendasi.

Alur data di berkas ini sengaja dipisah ke fungsi-fungsi kecil agar mudah dibaca:
1. MODEL_STRES dimuat sekali di memori saat server menyala (Load-Once Inference).
2. ambil_dan_validasi_input() memastikan seluruh input berupa angka >= 0 dan total <= 24 jam.
3. prediksi_tingkat_stres() menjalankan Random Forest Classifier pada DataFrame 5 kolom.
4. buat_rekomendasi() menerjemahkan kondisi harian menjadi panduan pemulihan (FR-04).
"""

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

# =====================================================================
# 1. KONFIGURASI FITUR, BATAS HARIAN, DAN PEMUATAN MODEL (1x SAAT STARTUP)
# =====================================================================
MAX_DAILY_HOURS = 24.0
ACTIVE_HOURS_THRESHOLD = 12.0

# Urutan spesifikasi fitur di bawah ini WAJIB identik dengan urutan saat training model.
FIELD_SPECS = (
    {
        "name": "Study_Hours_Per_Day",
        "label": "Belajar / Kerja Fokus",
        "hint": "Kuliah, tugas, atau pekerjaan layar fokus per hari",
        "placeholder": "Contoh: 7.5",
        "chart_label": "Belajar / Kerja",
        "color": "#181816",
        "is_active": True,
    },
    {
        "name": "Extracurricular_Hours_Per_Day",
        "label": "Ekstrakurikuler / Organisasi",
        "hint": "Kegiatan kepanitiaan, UKM, atau proyek tambahan",
        "placeholder": "Contoh: 2.0",
        "chart_label": "Ekstrakurikuler",
        "color": "#525866",
        "is_active": True,
    },
    {
        "name": "Sleep_Hours_Per_Day",
        "label": "Tidur / Istirahat Utama",
        "hint": "Total durasi tidur malam dan istirahat inti",
        "placeholder": "Contoh: 6.5",
        "chart_label": "Tidur",
        "color": "#2D6A4F",
        "is_active": False,
    },
    {
        "name": "Social_Hours_Per_Day",
        "label": "Interaksi Sosial",
        "hint": "Mengobrol santai bersama teman atau keluarga",
        "placeholder": "Contoh: 2.0",
        "chart_label": "Sosial",
        "color": "#C45D18",
        "is_active": True,
    },
    {
        "name": "Physical_Activity_Hours_Per_Day",
        "label": "Aktivitas Fisik / Olahraga",
        "hint": "Jalan kaki, peregangan, atau olahraga rutin",
        "placeholder": "Contoh: 1.5",
        "chart_label": "Aktivitas Fisik",
        "color": "#8C7A6B",
        "is_active": True,
    },
)

FEATURE_COLUMNS = [spec["name"] for spec in FIELD_SPECS]
ACTIVE_COLUMNS = [spec["name"] for spec in FIELD_SPECS if spec["is_active"]]

STRESS_UI_META = {
    "Low": {
        "label_id": "Rendah",
        "status_class": "status-low",
        "summary": "Ritme harian terjaga dengan tingkat beban mental yang masih terkendali.",
    },
    "Moderate": {
        "label_id": "Sedang",
        "status_class": "status-moderate",
        "summary": "Akumulasi beban mulai terasa; jeda pemulihan terstruktur sangat disarankan.",
    },
    "High": {
        "label_id": "Tinggi",
        "status_class": "status-high",
        "summary": "Risiko kelelahan dan burnout tinggi; prioritaskan pemulihan segera malam ini.",
    },
}

MODEL_PATH = Path(settings.BASE_DIR) / "prediction" / "ml_models" / "model_stres.joblib"
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File model ML tidak ditemukan di {MODEL_PATH}. "
        "Jalankan 'python train_model.py' terlebih dahulu."
    )

# Model dimuat satu kali di level modul agar inferensi per request < 10 ms.
MODEL_STRES = joblib.load(MODEL_PATH)


# =====================================================================
# 2. FUNGSI VALIDASI INPUT (FR-02: NUMERIK >= 0 DAN TOTAL <= 24 JAM)
# =====================================================================
def ambil_dan_validasi_input(
    post_data: Any,
) -> tuple[dict[str, float] | None, dict[str, str], str | None]:
    """Mengambil 5 input jam dari form POST dan memvalidasi kewajaran 24 jam.

    Mengembalikan tuple berisi:
    - data_jam: dictionary float tervalidasi (atau None jika gagal validasi)
    - raw_values: nilai string asli agar form tetap mempertahankan isian (sticky)
    - error_message: pesan galat bahasa Indonesia jika tidak valid
    """
    raw_values: dict[str, str] = {}
    data_jam: dict[str, float] = {}

    for spec in FIELD_SPECS:
        field_name = spec["name"]
        raw_text = str(post_data.get(field_name, "")).strip()
        raw_values[field_name] = raw_text

        if not raw_text:
            return None, raw_values, f"Kolom '{spec['label']}' wajib diisi."

        try:
            nilai = float(raw_text.replace(",", "."))
        except ValueError:
            return (
                None,
                raw_values,
                f"Kolom '{spec['label']}' harus berupa angka yang valid (contoh: 6.5).",
            )

        if not math.isfinite(nilai) or nilai < 0:
            return (
                None,
                raw_values,
                f"Kolom '{spec['label']}' tidak boleh bernilai negatif atau tidak valid.",
            )

        data_jam[field_name] = nilai

    total_jam_harian = sum(data_jam.values())
    if total_jam_harian > MAX_DAILY_HOURS:
        return (
            None,
            raw_values,
            (
                f"Total alokasi waktu Anda mencapai {total_jam_harian:.1f} jam, "
                f"melebihi batas wajar {int(MAX_DAILY_HOURS)} jam dalam sehari. "
                "Mohon sesuaikan kembali durasi aktivitas atau tidur Anda."
            ),
        )

    return data_jam, raw_values, None


# =====================================================================
# 3. FUNGSI PREDIKSI MACHINE LEARNING (FR-03)
# =====================================================================
def prediksi_tingkat_stres(data_jam: dict[str, float]) -> tuple[str, float]:
    """Menjalankan prediksi Random Forest dan menghitung skor keyakinan (%)."""
    # Konversi ke DataFrame dengan urutan kolom sesuai kontrak fitur saat training
    df_input = pd.DataFrame([data_jam], columns=FEATURE_COLUMNS)
    prediksi_kelas = str(MODEL_STRES.predict(df_input)[0])
    probabilitas_maks = float(MODEL_STRES.predict_proba(df_input)[0].max())
    confidence_score = round(probabilitas_maks * 100, 1)
    return prediksi_kelas, confidence_score


# =====================================================================
# 4. MESIN REKOMENDASI PEMULIHAN BERBASIS ATURAN (FR-04)
# =====================================================================
def buat_rekomendasi(
    data_jam: dict[str, float],
    total_aktivitas_aktif: float,
    level_stres: str,
) -> list[dict[str, str]]:
    """Menyusun panduan pemulihan personal berdasarkan pemicu spesifik pengguna."""
    aturan_pemulihan = (
        (
            total_aktivitas_aktif >= ACTIVE_HOURS_THRESHOLD,
            "Beban Harian Berlebih",
            "Hentikan pekerjaan berat; terapkan jeda bebas layar (screen-free) minimal 45 menit.",
        ),
        (
            data_jam["Study_Hours_Per_Day"] >= 8.0,
            "Kelelahan Kognitif",
            "Lakukan relaksasi mata (aturan 20-20-20) dan alihkan fokus ke aktivitas non-layar (membaca buku fisik / musik santai).",
        ),
        (
            data_jam["Physical_Activity_Hours_Per_Day"] < 1.0,
            "Kurang Gerak Fisik",
            "Lakukan peregangan ringan atau jalan santai 15–20 menit untuk menurunkan hormon kortisol.",
        ),
        (
            data_jam["Sleep_Hours_Per_Day"] < 6.5,
            "Defisit Pemulihan Tidur",
            "Targetkan tidur 7–8 jam malam ini; hindari kafein minimal 6 jam sebelum tidur.",
        ),
        (
            data_jam["Social_Hours_Per_Day"] < 1.0 and level_stres in ("Moderate", "High"),
            "Isolasi / Kejenuhan",
            "Luangkan waktu mengobrol santai bersama teman atau keluarga untuk pelepasan emosi.",
        ),
    )

    rekomendasi = [
        {"kategori": kategori, "saran": saran}
        for kondisi_terpenuhi, kategori, saran in aturan_pemulihan
        if kondisi_terpenuhi
    ]

    if not rekomendasi:
        rekomendasi.append(
            {
                "kategori": "Pola Seimbang",
                "saran": "Pertahankan keseimbangan ritme aktivitas-istirahat dan jaga hidrasi harian.",
            }
        )

    return rekomendasi


# =====================================================================
# 5. HELPER PENYUSUN DATA TAMPILAN & VIEW UTAMA
# =====================================================================
def _bangun_daftar_field(raw_values: dict[str, str]) -> list[dict[str, Any]]:
    """Menyiapkan daftar spesifikasi input beserta nilai sticky untuk template HTML."""
    return [
        {**spec, "value": raw_values.get(spec["name"], "")}
        for spec in FIELD_SPECS
    ]


def _bangun_data_grafik(data_jam: dict[str, float], total_24: float) -> dict[str, Any]:
    """Menyusun payload JSON aman untuk grafik donat proporsi 24 jam."""
    sisa_waktu = max(0.0, round(MAX_DAILY_HOURS - total_24, 2))
    labels = [spec["chart_label"] for spec in FIELD_SPECS] + ["Sisa Waktu"]
    values = [round(data_jam[spec["name"]], 2) for spec in FIELD_SPECS] + [sisa_waktu]
    colors = [spec["color"] for spec in FIELD_SPECS] + ["#E6E4DD"]
    rincian = [
        {
            "label": spec["chart_label"],
            "jam": round(data_jam[spec["name"]], 1),
            "persen": round((data_jam[spec["name"]] / MAX_DAILY_HOURS) * 100, 1),
            "color": spec["color"],
        }
        for spec in FIELD_SPECS
    ]
    rincian.append(
        {
            "label": "Sisa Waktu (Transisi / Luang)",
            "jam": round(sisa_waktu, 1),
            "persen": round((sisa_waktu / MAX_DAILY_HOURS) * 100, 1),
            "color": "#E6E4DD",
        }
    )
    return {"labels": labels, "values": values, "colors": colors, "rincian": rincian}


@require_http_methods(["GET", "POST"])
def index(request: HttpRequest) -> HttpResponse:
    """Menampilkan halaman utama dan memproses simulasi deteksi stres."""
    raw_values: dict[str, str] = {}
    context: dict[str, Any] = {
        "error_message": None,
        "hasil": None,
        "chart_data": None,
    }

    if request.method == "POST":
        data_jam, raw_values, error_message = ambil_dan_validasi_input(request.POST)

        if error_message is not None:
            context["error_message"] = error_message
        elif data_jam is not None:
            total_aktivitas = round(
                sum(data_jam[col] for col in ACTIVE_COLUMNS), 2
            )
            total_harian = round(sum(data_jam.values()), 2)
            level_stres, confidence_score = prediksi_tingkat_stres(data_jam)
            rekomendasi = buat_rekomendasi(data_jam, total_aktivitas, level_stres)
            ui_meta = STRESS_UI_META[level_stres]

            context["hasil"] = {
                "level_stres": level_stres,
                "label_id": ui_meta["label_id"],
                "status_class": ui_meta["status_class"],
                "summary": ui_meta["summary"],
                "confidence_score": confidence_score,
                "total_aktivitas": total_aktivitas,
                "total_harian": total_harian,
                "melewati_12_jam": total_aktivitas >= ACTIVE_HOURS_THRESHOLD,
                "rekomendasi": rekomendasi,
            }
            context["chart_data"] = _bangun_data_grafik(data_jam, total_harian)

    context["form_fields"] = _bangun_daftar_field(raw_values)
    return render(request, "prediction/index.html", context)
