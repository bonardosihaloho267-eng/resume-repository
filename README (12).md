# Studi Kasus 5 - Human-Computer Interaction & Data Visualization

**Narasumber:** Arafat Febriandirza (BRIN) | **Sesi 5** - 19 Oktober 2026

## Skenario
Tim kampus membuat dashboard layanan akademik. Studi kasus mencakup dua sisi materi.

### A. HCI - mengukur kegunaan dengan SUS
- Fungsi `sus_score()` menghitung **System Usability Scale** (10 pertanyaan, skor 0-100) sesuai aturan baku.
- Membandingkan antarmuka **lama vs baru** (30 responden simulasi per versi) dengan uji t Welch dan Cohen's d.
- Daftar 10 heuristik Nielsen disertakan di `hasil.json` untuk evaluasi heuristik.

### B. Visualisasi data
- Grafik **menyesatkan vs jujur** dari data yang sama (sumbu-y dipotong): kenaikan sebenarnya hanya **3,3%**, tetapi tampak dramatis pada grafik yang dipotong.
- Dashboard 4 panel: tren (line), perbandingan terurut (bar horizontal), kepuasan (bar dengan label nilai), sebaran waktu respons (box plot, bukan hanya rata-rata).
- Palet **Okabe-Ito** yang ramah buta warna.

## Menjalankan
```bash
python case_study.py
```
Keluaran: `outputs/01_sus.png`, `02_menyesatkan_vs_jujur.png`, `03_dashboard.png`, `hasil.json`.

## Hasil
- SUS antarmuka lama **55,6** (peringkat D) vs baru **75,8** (peringkat B); p < 0,001, Cohen's d = 3,28.
- Rata-rata industri SUS adalah sekitar 68 (garis putus-putus di grafik).

## Catatan
Respons SUS dan data layanan **simulasi**. Efek yang sangat besar (d = 3,28) sengaja dibuat untuk demonstrasi; pada pengujian pengguna nyata biasanya jauh lebih kecil. Untuk penelitian nyata, minta responden mengisi kuesioner SUS yang sama dan masukkan ke `sus_score()`.
