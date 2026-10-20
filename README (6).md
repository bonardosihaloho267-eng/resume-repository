# Studi Kasus 3 - Carrier (Career) and Competition in Industrial Era with GIS

**Narasumber:** Widi Prima Deva (Citycare Property, New Zealand) | **Sesi 3** - 15 Oktober 2026

## Skenario
Perusahaan manajemen properti mengelola 250 aset di Kota Bengkulu dengan 4 depot teknisi. Pertanyaan manajer:
1. Aset mana dilayani depot mana?
2. Aset mana yang kondisinya buruk **dan** berada di luar jangkauan depot (prioritas)?
3. Di mana depot baru sebaiknya dibangun?

## Kaitan dengan materi
| Keterampilan GIS yang disorot | Implementasi |
|---|---|
| Data spasial (lat/lon) | `asset`, `depot` sebagai titik koordinat |
| Pengukuran jarak | Rumus **haversine** (jarak lingkaran besar) |
| Analisis proximity / buffer | Radius layanan 3 km, persentase cakupan |
| Analisis lokasi (optimasi) | K-means **berbobot** kondisi aset untuk usulan depot baru |
| Visualisasi peta | Peta sebelum vs sesudah |
| Interoperabilitas | Ekspor `peta.geojson` (bisa dibuka di QGIS / geojson.io) |

Relevansi karier: ini contoh pekerjaan nyata seorang *GIS/spatial data analyst* di industri aset dan properti.

## Menjalankan
```bash
python case_study.py
```
Keluaran: `outputs/peta_cakupan.png`, `peta.geojson`, `aset_terlayani.csv`, `ringkasan.json`.

## Hasil
- Cakupan radius 3 km: **32,4% -> 46,0%** setelah satu depot baru di (-3,7937; 102,2903).
- 24 aset diprioritaskan (kondisi < 45 dan di luar radius).
- Jarak rata-rata ke depot terdekat turun dari 3,44 km ke 3,02 km.

## Catatan
- Koordinat aset dan depot **sintetis/ilustrasi**; jarak memakai garis lurus (bukan jarak jalan sebenarnya).
- Satu depot baru hanya menaikkan cakupan ke 46%, artinya jaringan masih butuh lebih banyak depot atau radius layanan perlu ditinjau. Ini justru insight yang berguna untuk diskusi.

## Pengembangan lanjutan
Pakai `geopandas` + `osmnx` untuk jarak jalan nyata, tambahkan `folium` untuk peta interaktif, dan optimasi lokasi dengan p-median.
