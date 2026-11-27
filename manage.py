#!/usr/bin/env python
"""Utilitas baris perintah Django untuk proyek StressGuard."""

import os
import sys


def main() -> None:
    """Menjalankan perintah manajemen Django."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stress_project.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django tidak ditemukan. Pastikan library Django sudah terpasang "
            "dan lingkungan Python yang digunakan sudah sesuai."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
