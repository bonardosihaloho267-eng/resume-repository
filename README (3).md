# Studi Kasus 2 - Bridging Global Infrastructure to Address Local Challenge with HPC

**Narasumber:** Asst. Prof. Dr. Worawan Diaz Carballo (Thammasat University) | **Sesi 2** - 5 Oktober 2026

## Skenario
Berapa peluang suatu DAS di Bengkulu banjir dalam hujan 7 hari? Dijawab dengan **simulasi Monte Carlo** jutaan skenario hujan. Makin banyak skenario, makin akurat estimasinya, sehingga komputasi paralel dibutuhkan.

## Kaitan dengan materi
| Konsep di materi | Implementasi |
|---|---|
| Komputasi paralel | `ProcessPoolExecutor`: skenario dibagi ke beberapa worker (data parallelism) |
| Masalah "embarrassingly parallel" | Tiap skenario saling bebas, jadi hampir tanpa komunikasi |
| Infrastruktur bersama / cluster | `job.slurm`: kode yang sama dijalankan di cluster HPC (SLURM) |
| Masalah lokal | Risiko banjir DAS di wilayah sendiri |
| Metrik HPC | Speedup dan efisiensi paralel dihitung otomatis |

Detail teknis: tiap worker memakai aliran acak independen (`SeedSequence.spawn`) supaya hasil tidak saling berkorelasi.

## Menjalankan
```bash
python case_study.py          # laptop / PC
sbatch job.slurm              # cluster HPC dengan SLURM (sesuaikan modul)
```
Keluaran: `outputs/hasil.json`, `outputs/hpc_hasil.png` (konvergensi + speedup).

## Hasil dari sandbox pembuatan (1 core)
- Estimasi peluang banjir dari 12 juta skenario: **sekitar 1,2%**.
- Konvergensi terlihat: estimasi stabil mulai sekitar 10^5 sampai 10^6 skenario.
- **Speedup tidak terlihat (sekitar 1x; berkisar 0,95 - 1,15x antar run karena fluktuasi waktu)** karena mesin pembuatan hanya punya 1 core. Hasil ini jujur apa adanya: pada mesin multi-core atau node cluster, angka speedup akan naik mendekati jumlah core. Silakan jalankan ulang di perangkat Anda dan masukkan angkanya ke laporan.
- Angka peluang berbeda sangat tipis antar jumlah worker (0,01197 - 0,01201) karena pembagian chunk dan seed berbeda; ini variasi Monte Carlo yang wajar.

## Catatan
Model hujan-limpasan di sini sangat disederhanakan (ilustrasi), bukan model hidrologi operasional.
