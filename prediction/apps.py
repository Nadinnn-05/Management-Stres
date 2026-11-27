"""Konfigurasi identitas aplikasi prediction."""

from django.apps import AppConfig


class PredictionConfig(AppConfig):
    """Mendaftarkan aplikasi prediction ke proyek Django."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "prediction"
