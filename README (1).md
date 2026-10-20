# Studi Kasus 1 - Make Sense of Data with Analysis and AI

**Narasumber:** Andria Arisal (BRIN) | **Sesi 1** - 5 Oktober 2026

## Skenario
Pemerintah kota ingin memahami data kualitas udara (PM2.5) per jam dan memprediksinya untuk peringatan dini.

## Kaitan dengan materi
| Konsep di materi | Implementasi di `case_study.py` |
|---|---|
| Data cleaning | Hapus duplikat, buang pencilan sensor (>500 ug/m3), interpolasi nilai kosong |
| Analisis deskriptif | Statistik ringkas + pola rata-rata PM2.5 per jam |
| Analisis diagnostik | Korelasi tiap variabel dengan PM2.5 |
| Analisis prediktif | Linear Regression vs Random Forest (split kronologis 80/20) |
| Analisis preskriptif | Fungsi `recommend()` mengubah prediksi menjadi tindakan |

## Menjalankan
```bash
pip install -r ../requirements.txt
python case_study.py
```
Keluaran di `outputs/`: `01_pola_harian.png`, `02_korelasi.png`, `03_prediksi.png`, `ringkasan.json`, `statistik_deskriptif.csv`.

## Hasil (seed tetap, dapat direproduksi)
- Data mentah 2.920 baris -> 2.880 setelah duplikat dibuang; 25 pencilan dibuang.
- PM2.5 rata-rata tertinggi pada **jam 18** (jam pulang kerja).
- Korelasi dengan PM2.5: lalu lintas **+0,81**, suhu +0,41, angin -0,40, kelembapan -0,31.
- Model: Linear Regression MAE 2,54 / R2 0,862; Random Forest MAE 2,71 / R2 0,844.

## Catatan penting
- Data **sintetis**. Hubungannya sengaja dibuat linear, sehingga Linear Regression sedikit mengungguli Random Forest. Pada data nyata yang non-linear, hasilnya bisa berbalik. Pelajarannya: model yang lebih kompleks tidak otomatis lebih baik.
- Korelasi tidak sama dengan sebab-akibat; analisis diagnostik hanya memberi petunjuk.
- Ambang tindakan di `recommend()` hanya ilustrasi dan perlu disesuaikan dengan standar resmi setempat.

## Pengembangan lanjutan
Ganti `generate_raw_data()` dengan data sensor/ISPU nyata, tambahkan fitur lag (PM2.5 jam sebelumnya), dan uji model deret waktu.
