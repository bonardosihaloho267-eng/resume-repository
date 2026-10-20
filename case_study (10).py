"""
Studi Kasus 5 - Human-Computer Interaction & Data Visualization
Narasumber: Arafat Febriandirza (BRIN)

Skenario : Tim pengembang membuat dashboard layanan akademik kampus. Studi kasus
           ini mencakup DUA sisi materi:
  (A) HCI            : mengukur kegunaan (usability) dengan kuesioner baku SUS
                       (System Usability Scale) pada DUA versi antarmuka, lalu
                       menguji apakah perbedaannya bermakna secara statistik.
  (B) Visualisasi    : memperlihatkan grafik yang MENYESATKAN vs grafik yang
                       JUJUR dari data yang sama, memakai palet ramah buta warna,
                       lalu menyusun dashboard 4 panel.
Catatan  : Respons SUS dan data layanan SINTETIS (untuk demonstrasi metode).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(11)

# Palet Okabe-Ito: dapat dibedakan oleh penyandang buta warna umum
OI = {"biru": "#0072B2", "oranye": "#E69F00", "hijau": "#009E73",
      "merah": "#D55E00", "abu": "#7F7F7F", "langit": "#56B4E9"}


# ------------------------------- (A) HCI: SUS -------------------------------
def sus_score(resp: np.ndarray) -> np.ndarray:
    """resp: (n_peserta, 10) skala 1-5. Item ganjil: (skor-1); genap: (5-skor).
    Jumlah dikali 2.5 -> skor 0-100."""
    resp = np.asarray(resp)
    odd = resp[:, 0::2] - 1
    even = 5 - resp[:, 1::2]
    return (odd.sum(axis=1) + even.sum(axis=1)) * 2.5


def simulate_sus(n: int, quality: float) -> np.ndarray:
    """quality 0-1: makin tinggi makin disukai. Item ganjil positif, genap negatif."""
    out = np.empty((n, 10), dtype=int)
    for j in range(10):
        base = 1 + 4 * quality if j % 2 == 0 else 5 - 4 * quality
        out[:, j] = np.clip(np.round(rng.normal(base, 0.8, n)), 1, 5)
    return out


def sus_grade(s: float) -> str:
    return ("A (sangat baik)" if s >= 80.3 else "B (baik)" if s >= 74 else
            "C (cukup)" if s >= 68 else "D (di bawah rata-rata)" if s >= 51 else "F (buruk)")


def hci_analysis():
    a = sus_score(simulate_sus(30, 0.55))     # antarmuka lama
    b = sus_score(simulate_sus(30, 0.80))     # antarmuka baru (hasil desain ulang)
    t, p = stats.ttest_ind(b, a, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (b.mean() - a.mean()) / pooled
    res = {"antarmuka_lama": {"rata2_SUS": round(a.mean(), 1), "peringkat": sus_grade(a.mean())},
           "antarmuka_baru": {"rata2_SUS": round(b.mean(), 1), "peringkat": sus_grade(b.mean())},
           "uji_t_welch": {"t": round(float(t), 2), "p": float(f"{p:.2e}"),
                           "cohens_d": round(float(d), 2)},
           "kesimpulan": "Perbedaan bermakna (p<0.05)" if p < .05 else "Tidak bermakna"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([a, b], tick_labels=["Antarmuka lama", "Antarmuka baru"], patch_artist=True,
                    widths=.5)
    for patch, c in zip(bp["boxes"], [OI["abu"], OI["biru"]]): patch.set_facecolor(c); patch.set_alpha(.7)
    ax.axhline(68, ls="--", color=OI["merah"]); ax.text(1.52, 69, "rata-rata industri = 68", color=OI["merah"], fontsize=8)
    ax.set(title="Skor SUS (n=30 per versi)", ylabel="Skor SUS (0-100)", ylim=(0, 100))
    ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(OUT / "01_sus.png", dpi=140); plt.close(fig)
    return res


# --------------------- (B) Visualisasi: menyesatkan vs jujur ----------------
def misleading_vs_honest():
    months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun"]
    tickets = np.array([1210, 1228, 1215, 1240, 1232, 1250])     # naik hanya ~3%
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(months, tickets, color=OI["merah"]); ax[0].set_ylim(1200, 1255)
    ax[0].set_title("MENYESATKAN: sumbu-y dipotong\n(terlihat naik drastis)")
    ax[1].bar(months, tickets, color=OI["biru"]); ax[1].set_ylim(0, 1400)
    ax[1].set_title("JUJUR: sumbu-y mulai dari 0\n(kenaikan kecil terlihat apa adanya)")
    for a_ in ax: a_.set_ylabel("Jumlah tiket layanan"); a_.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "02_menyesatkan_vs_jujur.png", dpi=140); plt.close(fig)
    return round((tickets[-1] / tickets[0] - 1) * 100, 1)


def dashboard():
    months = pd.date_range("2026-01-01", periods=12, freq="MS")
    svc = ["KRS Online", "Nilai", "Beasiswa", "Surat"]
    df = pd.DataFrame({s: (400 + 60 * i + 80 * np.sin(np.arange(12) / 2 + i)
                           + rng.normal(0, 25, 12)).clip(50) for i, s in enumerate(svc)},
                      index=months).round()
    sat = pd.Series([4.4, 4.1, 3.6, 3.9], index=svc)
    resp_time = {s: rng.gamma(2.5, 1.2 + .5 * i, 300) for i, s in enumerate(svc)}
    cols = [OI["biru"], OI["oranye"], OI["hijau"], OI["merah"]]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for s, c in zip(svc, cols):                                  # 1. tren
        ax[0, 0].plot(df.index, df[s], label=s, color=c, lw=2)
    ax[0, 0].set(title="Tren permintaan layanan (tiket/bulan)"); ax[0, 0].legend(fontsize=8, ncol=2)
    tot = df.sum().sort_values()                                 # 2. perbandingan (terurut)
    ax[0, 1].barh(tot.index, tot.values, color=[cols[svc.index(s)] for s in tot.index])
    ax[0, 1].set(title="Total tiket per layanan (terurut)")
    ax[1, 0].bar(sat.index, sat.values, color=cols); ax[1, 0].set(ylim=(0, 5), title="Kepuasan pengguna (skala 1-5)")
    for i, v in enumerate(sat.values): ax[1, 0].text(i, v + .08, f"{v:.1f}", ha="center", fontsize=9)
    ax[1, 1].boxplot(list(resp_time.values()), tick_labels=svc, showfliers=False, patch_artist=True,
                     boxprops=dict(facecolor=OI["langit"], alpha=.7))
    ax[1, 1].set(title="Waktu respons (jam) - sebaran, bukan hanya rata-rata")
    for a_ in ax.ravel():
        a_.grid(alpha=.25); a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Dashboard Layanan Akademik (data sintetis)", fontsize=14, fontweight="bold")
    import matplotlib.dates as mdates
    ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[0, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax[0, 1].set_xlabel("Jumlah tiket (12 bulan)")
    fig.tight_layout()
    fig.savefig(OUT / "03_dashboard.png", dpi=140); plt.close(fig)


CHECKLIST = [
    "Visibility of system status - pengguna selalu tahu apa yang sedang terjadi",
    "Match with real world - bahasa dan istilah akrab bagi pengguna",
    "User control & freedom - ada undo / batal",
    "Consistency & standards - elemen serupa berperilaku serupa",
    "Error prevention - cegah kesalahan sebelum terjadi",
    "Recognition over recall - jangan paksa pengguna mengingat",
    "Flexibility & efficiency - jalan pintas untuk pengguna mahir",
    "Aesthetic & minimalist design - hanya informasi yang relevan",
    "Help users recover from errors - pesan error jelas dan solutif",
    "Help & documentation - bantuan mudah dicari",
]


def main():
    hci = hci_analysis()
    pct = misleading_vs_honest()
    dashboard()
    res = {"HCI_SUS": hci, "kenaikan_tiket_Jan_Jun_persen": pct,
           "heuristik_nielsen_untuk_evaluasi": CHECKLIST}
    (OUT / "hasil.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "heuristik_nielsen_untuk_evaluasi"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
