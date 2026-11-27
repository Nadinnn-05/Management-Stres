"""Pengaturan utama Django untuk Sistem Deteksi & Manajemen Stres (StressGuard).

Proyek ini dirancang ringan tanpa ketergantungan basis data relasional maupun
sistem login, sesuai ruang lingkup MVP pada dokumen PRD.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Kunci keamanan untuk lingkungan lokal/pengembangan (override via env di produksi)
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "stressguard-local-dev-secret-key-uas-sistem-cerdas-2026",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() == "true"

ALLOWED_HOSTS = ["*"]

# Aplikasi yang aktif (cukup staticfiles dan app prediction)
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "prediction.apps.PredictionConfig",
]

# Middleware keamanan web dasar (termasuk proteksi CSRF untuk formulir POST)
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "stress_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "stress_project.wsgi.application"

# Aplikasi bersifat stateless (tanpa penyimpanan riwayat pengguna ke database)
DATABASES = {}

# Pengaturan bahasa Indonesia dan zona waktu WIB
LANGUAGE_CODE = "id"
TIME_ZONE = "Asia/Jakarta"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
