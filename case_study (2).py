"""
Studi Kasus 2 - Bridging Global Infrastructure to Address Local Challenge with HPC
Narasumber: Asst. Prof. Dr. Worawan Diaz Carballo (Thammasat University)

Skenario : Sebuah daerah aliran sungai (DAS) di Bengkulu ingin mengetahui
           PELUANG banjir: berapa persen skenario hujan 7 hari yang melebihi
           kapasitas tampung sungai? Jawabannya dihitung dengan simulasi Monte
           Carlo. Semakin banyak skenario, semakin akurat - sehingga komputasi
           paralel (inti dari HPC) sangat membantu.

Pola HPC : "embarrassingly parallel" - tiap skenario saling bebas, sehingga
           pekerjaan dapat dibagi ke banyak core/node (data parallelism).

File     : job.slurm berisi contoh cara menjalankan program yang sama di
           cluster HPC (infrastruktur bersama) tanpa mengubah kodenya.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# Parameter hidrologi sederhana (ilustrasi, bukan model hidrologi operasional)
DAYS = 7
SHAPE, SCALE = 0.6, 28.0        # hujan harian ~ Gamma (banyak hari ringan, sesekali ekstrem)
WET_PROB = 0.55                 # peluang suatu hari hujan
RUNOFF_COEF = 0.45              # fraksi hujan yang menjadi limpasan
CATCHMENT_MM_CAPACITY = 95.0    # kapasitas tampung sungai (mm limpasan setara)


def simulate_chunk(args):
    """Menjalankan n skenario dan mengembalikan jumlah skenario yang banjir."""
    n, seed_seq = args
    rng = np.random.default_rng(seed_seq)            # aliran acak independen per worker
    rain = rng.gamma(SHAPE, SCALE, size=(n, DAYS))
    rain *= rng.random((n, DAYS)) < WET_PROB
    runoff = RUNOFF_COEF * rain.sum(axis=1)
    return int((runoff > CATCHMENT_MM_CAPACITY).sum()), n


def run(n_total: int, workers: int, seed: int = 2026):
    n_chunks = max(workers * 4, 4)                   # banyak chunk kecil -> beban seimbang
    sizes = [n_total // n_chunks] * n_chunks
    sizes[-1] += n_total - sum(sizes)
    seeds = np.random.SeedSequence(seed).spawn(n_chunks)
    t0 = time.perf_counter()
    if workers == 1:
        parts = [simulate_chunk(a) for a in zip(sizes, seeds)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(simulate_chunk, zip(sizes, seeds)))
    dt = time.perf_counter() - t0
    floods = sum(p[0] for p in parts); total = sum(p[1] for p in parts)
    return floods / total, dt


def main():
    cores = os.cpu_count() or 1
    print(f"Jumlah core terdeteksi: {cores}")
    n_total = 12_000_000
    worker_list = sorted({1, 2, 4, cores})
    rows = []
    for w in worker_list:
        p, dt = run(n_total, w)
        rows.append({"workers": w, "peluang_banjir": round(p, 5), "waktu_detik": round(dt, 3)})
        print(f"workers={w:>2}  P(banjir)={p:.5f}  waktu={dt:.2f}s")
    base = rows[0]["waktu_detik"]
    for r in rows:
        r["speedup"] = round(base / r["waktu_detik"], 2)
        r["efisiensi"] = round(r["speedup"] / r["workers"], 2)

    # Konvergensi: makin banyak skenario, estimasi makin stabil
    ns = [10**k for k in range(2, 8)]
    est = [run(n, 1, seed=7)[0] for n in ns]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(ns, est, marker="o", color="#1F3A5F")
    ax[0].axhline(rows[0]["peluang_banjir"], ls="--", color="gray", label="estimasi 12 juta skenario")
    ax[0].set(title="Konvergensi estimasi peluang banjir", xlabel="Jumlah skenario", ylabel="P(banjir)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    w = [r["workers"] for r in rows]
    ax[1].plot(w, [r["speedup"] for r in rows], marker="o", label="Speedup aktual", color="#C0392B")
    ax[1].plot(w, w, ls="--", color="gray", label="Speedup ideal")
    ax[1].set(title=f"Speedup paralel (mesin ini: {cores} core)", xlabel="Jumlah worker", ylabel="Speedup")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "hpc_hasil.png", dpi=140); plt.close(fig)

    result = {"core_terdeteksi": cores, "jumlah_skenario": n_total, "hasil": rows,
              "catatan": "Speedup hanya terlihat bila mesin memiliki >1 core. "
                         "Di cluster HPC, jalankan lewat job.slurm."}
    (OUT / "hasil.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
