# Studi Kasus 6 - Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI

**Narasumber:** Bart Slaghter (Wageningen University & Research) | **Sesi 6** - 19 Oktober 2026

## Skenario
Sistem peringatan deforestasi menunjukkan **di mana** hutan hilang, bukan **mengapa**. Studi kasus ini melatih model untuk menentukan **penyebab langsung** tiap alert: pertanian skala kecil, perkebunan, penebangan, pertambangan, atau kebakaran.

## Kaitan dengan materi
| Konsep di materi | Implementasi |
|---|---|
| Fitur dari citra satelit | NDVI sebelum/sesudah, penurunan NBR (indikator api), tekstur |
| Bentuk & konteks lokasi | Luas, kelonjongan poligon, jarak ke jalan dan sungai |
| Klasifikasi penyebab dengan AI | Random Forest (5 kelas, `class_weight="balanced"`) |
| Tantangan: tutupan awan | Fitur spektral dihilangkan acak 0-70%, lalu uji ketahanan model |
| Mitigasi | Augmentasi data berawan saat pelatihan |
| Penggunaan operasional | Tabel prediksi + tingkat keyakinan + penanda "perlu verifikasi lapangan" |

## Menjalankan
```bash
python case_study.py
```
Keluaran: `outputs/01_confusion_matrix.png`, `02_feature_importance.png`, `03_ketahanan_awan.png`, `contoh_prediksi_alert_baru.csv`, `hasil.json`.

## Hasil (data simulasi)
- Akurasi uji **0,943**, macro F1 0,943. Kelas tersulit: pertambangan (F1 0,90) dan perkebunan (0,92); termudah: kebakaran (0,99).
- Fitur terpenting: `nbr_drop`, `ndvi_after`, `area_ha`, `elongation`.
- Ketahanan awan (macro F1): pada 70% data berawan, model biasa turun ke **0,64**, sedangkan model dengan augmentasi awan bertahan di **0,76**.

## Peringatan penting
**Tidak ada citra satelit nyata** di repositori ini. Fitur disimulasikan dengan distribusi yang masuk akal, sehingga angka akurasi **tidak boleh** dianggap performa dunia nyata. Studi kasus ini mendemonstrasikan *alur kerja*. Untuk data nyata, ganti `generate_alerts()` dengan ekstraksi fitur dari Sentinel-2/Landsat (mis. via Google Earth Engine atau `rasterio`) dan label hasil verifikasi pakar.
