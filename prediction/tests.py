"""Pengujian otomatis untuk aplikasi StressGuard (halaman, ML, aturan, dan validasi)."""

from django.test import SimpleTestCase
from django.urls import reverse


class StressGuardViewsTests(SimpleTestCase):
    """Menguji seluruh alur fungsional aplikasi web StressGuard."""

    def setUp(self) -> None:
        self.url = reverse("prediction:index")

    def test_halaman_utama_dapat_diakses(self) -> None:
        """Memastikan request GET mengembalikan status 200 dan judul aplikasi."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "StressGuard")
        self.assertContains(response, "Study_Hours_Per_Day")

    def test_prediksi_stres_rendah_pola_seimbang(self) -> None:
        """Menguji skenario seimbang (< 12 jam aktif) menghasilkan Low dan Pola Seimbang."""
        payload = {
            "Study_Hours_Per_Day": "5.0",
            "Extracurricular_Hours_Per_Day": "1.5",
            "Sleep_Hours_Per_Day": "8.0",
            "Social_Hours_Per_Day": "2.5",
            "Physical_Activity_Hours_Per_Day": "1.5",
        }
        response = self.client.post(self.url, data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["error_message"])
        hasil = response.context["hasil"]
        self.assertEqual(hasil["level_stres"], "Low")
        self.assertFalse(hasil["melewati_12_jam"])
        self.assertEqual(hasil["rekomendasi"][0]["kategori"], "Pola Seimbang")

    def test_prediksi_stres_sedang_dan_isolasi(self) -> None:
        """Menguji skenario beban moderat + minim sosial memicu Isolasi / Kejenuhan."""
        payload = {
            "Study_Hours_Per_Day": "7.0",
            "Extracurricular_Hours_Per_Day": "3.0",
            "Sleep_Hours_Per_Day": "7.0",
            "Social_Hours_Per_Day": "0.5",
            "Physical_Activity_Hours_Per_Day": "2.0",
        }
        response = self.client.post(self.url, data=payload)
        self.assertEqual(response.status_code, 200)
        hasil = response.context["hasil"]
        self.assertEqual(hasil["level_stres"], "Moderate")
        self.assertTrue(hasil["melewati_12_jam"])
        kategori_list = [item["kategori"] for item in hasil["rekomendasi"]]
        self.assertIn("Beban Harian Berlebih", kategori_list)
        self.assertIn("Isolasi / Kejenuhan", kategori_list)

    def test_prediksi_stres_tinggi_skenario_ekstrem(self) -> None:
        """Menguji skenario padat 14 jam dan kurang tidur menghasilkan High + 4 aturan."""
        payload = {
            "Study_Hours_Per_Day": "9.5",
            "Extracurricular_Hours_Per_Day": "2.5",
            "Sleep_Hours_Per_Day": "5.0",
            "Social_Hours_Per_Day": "1.5",
            "Physical_Activity_Hours_Per_Day": "0.5",
        }
        response = self.client.post(self.url, data=payload)
        self.assertEqual(response.status_code, 200)
        hasil = response.context["hasil"]
        self.assertEqual(hasil["level_stres"], "High")
        self.assertTrue(hasil["melewati_12_jam"])
        kategori_list = [item["kategori"] for item in hasil["rekomendasi"]]
        self.assertIn("Beban Harian Berlebih", kategori_list)
        self.assertIn("Kelelahan Kognitif", kategori_list)
        self.assertIn("Kurang Gerak Fisik", kategori_list)
        self.assertIn("Defisit Pemulihan Tidur", kategori_list)
        self.assertContains(response, 'style="width: 100.0%;"')
        self.assertContains(response, "100.0%")

    def test_validasi_total_jam_melebihi_24_jam_ditolak(self) -> None:
        """Memastikan total waktu > 24 jam ditolak dan menjaga nilai sticky form."""
        payload = {
            "Study_Hours_Per_Day": "12.0",
            "Extracurricular_Hours_Per_Day": "4.0",
            "Sleep_Hours_Per_Day": "6.0",
            "Social_Hours_Per_Day": "2.0",
            "Physical_Activity_Hours_Per_Day": "2.0",
        }
        response = self.client.post(self.url, data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["hasil"])
        self.assertIsNotNone(response.context["error_message"])
        self.assertIn("24 jam", response.context["error_message"])
        # Pastikan input tetap bertahan (sticky) di form
        self.assertContains(response, 'value="12.0"')

    def test_validasi_input_negatif_atau_teks_ditolak(self) -> None:
        """Memastikan input bukan angka atau nilai negatif tidak menyebabkan error 500."""
        payload_teks = {
            "Study_Hours_Per_Day": "delapan",
            "Extracurricular_Hours_Per_Day": "2.0",
            "Sleep_Hours_Per_Day": "7.0",
            "Social_Hours_Per_Day": "2.0",
            "Physical_Activity_Hours_Per_Day": "1.0",
        }
        res_teks = self.client.post(self.url, data=payload_teks)
        self.assertEqual(res_teks.status_code, 200)
        self.assertIsNone(res_teks.context["hasil"])
        self.assertIsNotNone(res_teks.context["error_message"])

        payload_negatif = {**payload_teks, "Study_Hours_Per_Day": "-1.5"}
        res_neg = self.client.post(self.url, data=payload_negatif)
        self.assertEqual(res_neg.status_code, 200)
        self.assertIsNone(res_neg.context["hasil"])
        self.assertIsNotNone(res_neg.context["error_message"])
