# StressGuard — Sistem Deteksi & Manajemen Stres Pasca-Aktivitas Harian

Aplikasi web berbasis **Django** dan **Machine Learning (Random Forest Classifier)** untuk mengevaluasi tingkat risiko stres (`Low`, `Moderate`, `High`) setelah menjalani aktivitas harian yang padat (khususnya beban aktif $\ge 12$ jam/hari) serta memberikan rekomendasi pemulihan yang dipersonalisasi.

> **Konteks Akademik:** Proyek Akhir Individu (UAS) Mata Kuliah **Sistem Cerdas (TI24317)**, Program Studi Teknologi Informasi, Politeknik Negeri Madiun.

---

## 1. Cara Kerja Sistem (Untuk Orang Awam)

Aplikasi ini bekerja layaknya **klinik konsultasi digital** dengan alur yang sederhana:

1. **Pengguna mengisi 5 durasi aktivitas harian** (Belajar/Kerja Fokus, Ekstrakurikuler, Tidur, Interaksi Sosial, dan Aktivitas Fisik) dalam satuan jam.
2. **Validasi Batas Harian (`24` Jam):** Sistem memastikan seluruh input berupa angka $\ge 0$ dan total waktu tidak melebihi $24$ jam dalam sehari.
3. **Otak AI (*Random Forest Classifier*):** Model membaca pola 5 jam tersebut dan memprediksi tingkat stres (`Low`, `Moderate`, atau `High`) beserta persentase keyakinannya (*confidence score*).
4. **Mesin Rekomendasi Pemulihan (*Rule-Based*):** Sistem memeriksa pemicu kelelahan spesifik (misalnya beban aktif $\ge 12$ jam, belajar $\ge 8$ jam, tidur $< 6.5$ jam, atau kurang olahraga $< 1$ jam) dan menampilkan langkah pemulihan praktis.
5. **Visualisasi 24 Jam:** Menampilkan grafik donat interaktif untuk mengevaluasi keseimbangan waktu harian.

---

## 2. Struktur Direktori Proyek

```text
Management Stres/
├── Makefile                              # Pintasan perintah (run, test, check, train, clean)
├── manage.py                             # Perintah utama untuk menjalankan server & test Django
├── requirements.txt                      # Daftar library Python dan versi yang digunakan
├── README.md                             # Dokumentasi proyek & laporan evaluasi model
├── train_model.py                        # Skrip otomatis untuk melatih model & membuat grafik
├── notebook_training_stres.ipynb         # Notebook interaktif untuk eksplorasi & pelatihan model
├── evaluasi_model.png                    # Grafik Confusion Matrix & Feature Importance
├── student_lifestyle_dataset.csv         # Dataset terbuka gaya hidup mahasiswa (2.000 baris)
│
├── stress_project/                       # Pusat konfigurasi proyek Django
│   ├── settings.py                       # Pengaturan aplikasi, bahasa, zona waktu (WIB)
│   ├── urls.py                           # Pintu gerbang rute URL utama
│   └── wsgi.py                           # Konfigurasi standar server web Python
│
└── prediction/                           # Aplikasi utama prediksi & manajemen stres
    ├── apps.py                           # Identitas app 'prediction'
    ├── urls.py                           # Pemetaan rute '/' ke halaman utama
    ├── views.py                          # Logika validasi, prediksi ML, dan mesin rekomendasi
    ├── tests.py                          # Pengujian otomatis (6 skenario uji)
    ├── ml_models/
    │   └── model_stres.joblib            # Model Random Forest terlatih (dimuat 1x saat startup)
    └── templates/
        └── prediction/
            └── index.html                # Antarmuka web Modern Minimalis + Chart.js
```

---

## 3. Informasi Dataset & Seleksi Fitur

* **Nama Dataset:** Student Lifestyle Dataset
* **Sumber Kaggle:** `https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset`
* **Lisensi:** Tercatat sebagai **Apache License 2.0** pada indeks/cermin OpenML (verifikasi ulang langsung di halaman Kaggle sebelum pengumpulan final).
* **Ukuran:** $2.000$ baris observasi, $8$ kolom ($0$ *missing values*, $0$ baris duplikat).
* **Distribusi Target (`Stress_Level`):** `High` ($1.029$), `Moderate` ($674$), `Low` ($297$).

### Fitur yang Digunakan vs Dibuang
| Kolom | Status | Alasan |
| :--- | :--- | :--- |
| `Study_Hours_Per_Day` | **Dipakai ($X_1$)** | Mengukur beban fokus/kognitif harian |
| `Extracurricular_Hours_Per_Day` | **Dipakai ($X_2$)** | Mengukur beban organisasi/kegiatan tambahan |
| `Sleep_Hours_Per_Day` | **Dipakai ($X_3$)** | Mengukur durasi pemulihan biologis utama |
| `Social_Hours_Per_Day` | **Dipakai ($X_4$)** | Mengukur pelepasan emosi melalui interaksi sosial |
| `Physical_Activity_Hours_Per_Day` | **Dipakai ($X_5$)** | Mengukur aktivitas fisik penurun stres |
| `Student_ID` | **Dibuang** | Hanya nomor identitas unik; tidak memiliki pola prediktif |
| `GPA` | **Dibuang** | Mengukur prestasi kumulatif, bukan kelelahan harian pasca-aktivitas |

---

## 4. Hasil Evaluasi Model Machine Learning

Model dilatih menggunakan `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)` dengan pembagian data terstratifikasi $80\%$ data latih ($1.600$ baris) dan $20\%$ data uji ($400$ baris).

| Metrik Evaluasi | Hasil Terukur |
| :--- | :--- |
| **Akurasi Data Latih** | `100.00%` |
| **Akurasi Data Uji** | `100.00%` |
| **Overfitting Gap** | `0.00%` |

### Classification Report (Data Uji — 400 Sampel)
| Kelas | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `Low` | `1.0000` | `1.0000` | `1.0000` | `59` |
| `Moderate` | `1.0000` | `1.0000` | `1.0000` | `135` |
| `High` | `1.0000` | `1.0000` | `1.0000` | `206` |

### Tingkat Pengaruh Fitur (*Feature Importance*)
1. `Study_Hours_Per_Day`: `0.6720`
2. `Sleep_Hours_Per_Day`: `0.2317`
3. `Physical_Activity_Hours_Per_Day`: `0.0572`
4. `Social_Hours_Per_Day`: `0.0256`
5. `Extracurricular_Hours_Per_Day`: `0.0136`

Grafik *Confusion Matrix* dan *Feature Importance* tersimpan di [evaluasi_model.png](evaluasi_model.png).

---

## 5. Cara Menjalankan Aplikasi Secara Lokal

### Opsi A: Menggunakan `Makefile`
*(Catatan Windows: jika perintah `make` belum dikenali di PowerShell, gunakan `mingw32-make` atau buat alias `Set-Alias make mingw32-make`)*

1. Lihat semua perintah yang tersedia:
   ```powershell
   make help
   ```
2. Jalankan server pengembangan Django di `http://127.0.0.1:8000/`:
   ```powershell
   make run
   ```
3. Perintah berguna lainnya:
   * `make test` — Menjalankan 6 unit test otomatis.
   * `make check` — Memeriksa konfigurasi proyek Django.
   * `make train` — Melatih ulang model Random Forest dan memperbarui `evaluasi_model.png`.
   * `make clean` — Menghapus folder cache `__pycache__`.

### Opsi B: Perintah Python Manual
1. Pastikan Python dan pustaka pada `requirements.txt` sudah terpasang:
   ```powershell
   pip install -r requirements.txt
   ```
2. *(Opsional)* Latih ulang model dan buat grafik evaluasi:
   ```powershell
   python train_model.py
   ```
3. Jalankan pengujian otomatis untuk memastikan seluruh fitur berjalan normal:
   ```powershell
   python manage.py test prediction
   ```
4. Jalankan server pengembangan lokal Django:
   ```powershell
   python manage.py runserver
   ```
5. Buka browser dan kunjungi `http://127.0.0.1:8000/`.

---

## 6. Refleksi Jujur, Keterbatasan Model, & Etika AI

* **Interpretasi Akurasi 100%:** Dataset *Student Lifestyle* dari Kaggle memiliki batas pemisahan pola yang sangat tegas dan terstruktur antar kelas (terutama pada fitur jam belajar dan jam tidur), sehingga `RandomForestClassifier` mampu mencapai akurasi $100\%$ pada data uji. Di dunia nyata yang penuh variasi psikologis dan faktor tak terukur, tingkat akurasi tidak akan selalu $100\%$.
* **Keterbatasan Domain:** Model ini hanya mempertimbangkan durasi waktu (kuantitas jam), belum mencakup kualitas tidur, tekanan emosional mendadak, kondisi finansial, atau faktor klinis lainnya.
* **Disclaimer & Privasi:** Aplikasi ini berfungsi sebagai alat bantu refleksi gaya hidup, **bukan pengganti psikolog atau diagnosis medis profesional**, serta tidak menyimpan data identitas pribadi (*PII*) pengguna ke dalam basis data mana pun.
