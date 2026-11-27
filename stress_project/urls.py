"""Pemetaan URL utama proyek StressGuard."""

from django.urls import include, path

urlpatterns = [
    path("", include("prediction.urls")),
]
