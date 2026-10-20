# Studi Kasus 4 - Introduction to Machine Learning

**Narasumber:** Dr. Taniza Tajuddin (UiTM Cawangan Kedah, Malaysia) | **Sesi 4** - 16 Oktober 2026

## Skenario
Laboratorium mengklasifikasikan 3 jenis anggur dari 13 hasil uji kimia (dataset `wine` bawaan scikit-learn, tanpa unduhan).

## Kaitan dengan materi
| Konsep di materi | Implementasi |
|---|---|
| Alur kerja ML | split data -> standarisasi -> latih -> evaluasi |
| Supervised learning | 4 algoritma klasifikasi dibandingkan dengan 5-fold cross-validation |
| Overfitting / underfitting | Kurva akurasi latih vs validasi terhadap kedalaman Decision Tree |
| Unsupervised learning | K-means tanpa label, dicek dengan Adjusted Rand Index |
| Evaluasi | Confusion matrix, precision / recall / F1 |

## Menjalankan
```bash
python case_study.py
```
Keluaran: `outputs/01_confusion_matrix.png`, `02_overfitting.png`, `03_kmeans.png`, `hasil.json`.

## Hasil
| Model | Akurasi CV (data latih) |
|---|---|
| Random Forest | 0,985 |
| Logistic Regression | 0,977 |
| KNN (k=5) | 0,955 |
| Decision Tree | 0,903 |

- Model terpilih: Random Forest, akurasi data uji **1,00**.
- Decision Tree kedalaman 12: akurasi latih 1,00 tetapi validasi 0,90 (**overfitting**); validasi terbaik pada kedalaman 3 (0,91).
- K-means tanpa label menemukan kelompok yang sangat mirip kultivar asli (ARI **0,90**).

## Catatan
Data uji hanya 45 sampel, jadi akurasi 1,00 tidak berarti model sempurna; selisih antar model teratas kecil dan bisa berubah dengan seed lain. Karena itu pemilihan model dilakukan dengan cross-validation, bukan data uji.
