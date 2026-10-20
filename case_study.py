"""
Studi Kasus 1 - Make Sense of Data with Analysis and AI
Narasumber: Andria Arisal (BRIN)

Skenario : Pemerintah kota ingin memahami data kualitas udara (PM2.5) per jam
           dan memprediksinya agar dapat memberi peringatan dini.
Catatan  : Data bersifat SINTETIS (dibangkitkan program) supaya dapat dijalankan
           tanpa unduhan apa pun. Struktur masalahnya meniru data nyata:
           ada nilai kosong, data ganda, dan pencilan.

Alur     : 1) bangkitkan data -> 2) cleaning -> 3) analisis deskriptif
           4) analisis diagnostik -> 5) prediktif (ML) -> 6) preskriptif
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


# 1) DATA MENTAH -------------------------------------------------------------
def generate_raw_data(days: int = 120) -> pd.DataFrame:
    t = pd.date_range("2026-06-01", periods=days * 24, freq="h")
    hour = t.hour.to_numpy()
    traffic = (
        40
        + 45 * np.exp(-((hour - 7.5) ** 2) / 4)     # jam sibuk pagi
        + 55 * np.exp(-((hour - 17.5) ** 2) / 5)    # jam sibuk sore
        + rng.normal(0, 5, len(t))
    ).clip(0)
    temp = 27 + 4 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.8, len(t))
    humidity = (80 - 2.2 * (temp - 27) + rng.normal(0, 4, len(t))).clip(30, 100)
    wind = np.abs(rng.normal(2.5, 1.2, len(t)))
    pm25 = (
        12 + 0.35 * traffic - 3.2 * wind + 0.15 * humidity
        + 0.8 * (temp - 27) + rng.normal(0, 3, len(t))
    ).clip(1)
    df = pd.DataFrame({
        "timestamp": t, "traffic_index": traffic, "temp_c": temp,
        "humidity_pct": humidity, "wind_ms": wind, "pm25": pm25,
    })
    # --- merusak data agar mirip kondisi nyata ---
    for col in ["traffic_index", "temp_c", "humidity_pct", "wind_ms", "pm25"]:
        df.loc[rng.choice(len(df), int(0.03 * len(df)), replace=False), col] = np.nan
    outl = rng.choice(len(df), 25, replace=False)
    df.loc[outl, "pm25"] = rng.uniform(600, 999, len(outl))      # sensor error
    df = pd.concat([df, df.sample(40, random_state=1)], ignore_index=True)  # duplikat
    return df.sample(frac=1, random_state=3).reset_index(drop=True)


# 2) CLEANING ----------------------------------------------------------------
def clean(df: pd.DataFrame):
    report = {"baris_awal": int(len(df))}
    df = df.drop_duplicates().copy()
    report["setelah_hapus_duplikat"] = int(len(df))
    df = df.sort_values("timestamp").set_index("timestamp")
    report["nilai_kosong_per_kolom"] = {k: int(v) for k, v in df.isna().sum().items()}
    # pencilan: PM2.5 > 500 ug/m3 tidak masuk akal -> dianggap error sensor
    n_out = int((df["pm25"] > 500).sum())
    df.loc[df["pm25"] > 500, "pm25"] = np.nan
    report["pencilan_pm25_dibuang"] = n_out
    df = df.interpolate(method="time", limit_direction="both")
    report["baris_akhir"] = int(len(df))
    return df, report


# 3) DESKRIPTIF --------------------------------------------------------------
def descriptive(df: pd.DataFrame):
    stats = df.describe().round(2)
    stats.to_csv(OUT / "statistik_deskriptif.csv")
    hourly = df.groupby(df.index.hour)["pm25"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hourly.index, hourly.values, marker="o", color="#1F3A5F")
    ax.set(title="Rata-rata PM2.5 menurut jam", xlabel="Jam (WIB)", ylabel="PM2.5 (ug/m3)")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "01_pola_harian.png", dpi=140)
    plt.close(fig)
    return hourly


# 4) DIAGNOSTIK --------------------------------------------------------------
def diagnostic(df: pd.DataFrame):
    corr = df.corr()["pm25"].drop("pm25").sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = ["#C0392B" if v > 0 else "#2E86C1" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set(title="Korelasi tiap variabel dengan PM2.5", xlabel="Koefisien korelasi Pearson")
    ax.axvline(0, color="k", lw=.8); fig.tight_layout()
    fig.savefig(OUT / "02_korelasi.png", dpi=140); plt.close(fig)
    return corr


# 5) PREDIKTIF ---------------------------------------------------------------
def predictive(df: pd.DataFrame):
    X = df[["traffic_index", "temp_c", "humidity_pct", "wind_ms"]].copy()
    X["hour"] = df.index.hour
    y = df["pm25"]
    split = int(len(df) * 0.8)            # split kronologis, bukan acak (data deret waktu)
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                               random_state=0, n_jobs=-1),
    }
    res, preds = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr); p = m.predict(Xte); preds[name] = p
        res[name] = {"MAE": round(mean_absolute_error(yte, p), 3),
                     "R2": round(r2_score(yte, p), 3)}
    fig, ax = plt.subplots(figsize=(9, 4))
    seg = slice(0, 24 * 5)
    ax.plot(yte.index[seg], yte.values[seg], label="Aktual", color="k", lw=1.2)
    ax.plot(yte.index[seg], preds["Random Forest"][seg], label="Prediksi (RF)",
            color="#E67E22", lw=1.2)
    ax.set(title="Prediksi vs aktual PM2.5 (5 hari pertama data uji)", ylabel="PM2.5 (ug/m3)")
    ax.legend(); ax.grid(alpha=.3); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / "03_prediksi.png", dpi=140); plt.close(fig)
    imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
    return res, imp.sort_values(ascending=False).round(3), yte, preds["Random Forest"]


# 6) PRESKRIPTIF -------------------------------------------------------------
def recommend(pm25: float) -> str:
    if pm25 < 15.5:  return "Baik - tidak ada tindakan."
    if pm25 < 55.4:  return "Sedang - kelompok sensitif kurangi aktivitas luar."
    return "Tidak sehat - batasi lalu lintas, sebarkan peringatan publik."


def main():
    raw = generate_raw_data()
    df, rep = clean(raw)
    hourly = descriptive(df)
    corr = diagnostic(df)
    res, imp, yte, p = predictive(df)

    summary = {
        "cleaning": rep,
        "jam_PM25_tertinggi": int(hourly.idxmax()),
        "korelasi_dengan_pm25": corr.round(3).to_dict(),
        "performa_model": res,
        "pentingnya_fitur_RF": imp.to_dict(),
        "contoh_rekomendasi": [
            {"prediksi_pm25": round(float(v), 1), "tindakan": recommend(v)}
            for v in (p[0], p[len(p) // 2], float(np.max(p)))
        ],
    }
    (OUT / "ringkasan.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
