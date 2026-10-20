"""
Studi Kasus 6 - Attributing Direct Drivers to Deforestation Alerts with
                Satellite Imagery and AI
Narasumber: Bart Slaghter (Wageningen University & Research)

Skenario : Sistem peringatan deforestasi menghasilkan ribuan "alert" (poligon
           hilangnya hutan) tetapi tidak menjelaskan PENYEBAB-nya. Kita latih
           model untuk mengklasifikasikan penyebab langsung tiap alert dari
           ciri-ciri yang diekstrak dari citra satelit.

PENTING  : Tidak ada citra satelit asli di sini. Ciri tiap alert DISIMULASIKAN
           dengan distribusi yang masuk akal secara fisik (misal kebakaran =>
           penurunan NBR besar; pertambangan => NDVI jatuh sangat rendah dan
           bentuk tidak beraturan). Tujuannya mendemonstrasikan ALUR KERJA
           (fitur -> model -> evaluasi -> ketahanan terhadap awan), bukan
           menghasilkan angka akurasi untuk dunia nyata. Untuk data nyata,
           ganti fungsi generate_alerts() dengan ekstraksi fitur dari
           Sentinel-2/Landsat (mis. via Google Earth Engine / rasterio).

Fitur per alert (semua diturunkan dari citra):
  ndvi_before, ndvi_after, ndvi_drop  : tutupan vegetasi sebelum/sesudah
  nbr_drop                            : Normalized Burn Ratio turun (indikator api)
  area_ha, elongation                 : luas dan kelonjongan bentuk poligon
  dist_road_km, dist_river_km         : kedekatan ke jalan dan sungai
  texture                             : variasi spasial NDVI setelah kejadian
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import train_test_split

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
SEED = 2026
rng = np.random.default_rng(SEED)

CLASSES = ["Pertanian skala kecil", "Perkebunan", "Penebangan", "Pertambangan", "Kebakaran"]
FEATURES = ["ndvi_before", "ndvi_after", "ndvi_drop", "nbr_drop", "area_ha",
            "elongation", "dist_road_km", "dist_river_km", "texture"]

# (rata2, simpangan) per fitur dan kelas -- kelas sengaja tumpang-tindih
# urutan fitur: ndvi_before, ndvi_after, nbr_drop, log_area, elongation, road, river, texture
PROFILE = {
    "Pertanian skala kecil": [(.78, .07), (.35, .10), (.10, .06), (0.7, .6), (1.6, .5), (2.5, 1.5), (1.5, 1.0), (.14, .05)],
    "Perkebunan":            [(.82, .06), (.30, .09), (.12, .06), (2.9, .7), (1.3, .3), (3.5, 2.0), (3.0, 2.0), (.07, .03)],
    "Penebangan":            [(.84, .05), (.55, .10), (.08, .05), (1.8, .7), (3.2, 1.0), (1.2, .9), (3.5, 2.0), (.18, .06)],
    "Pertambangan":          [(.76, .08), (.12, .07), (.09, .06), (2.2, .8), (1.8, .6), (2.0, 1.5), (1.2, .9), (.10, .05)],
    "Kebakaran":             [(.80, .07), (.38, .12), (.45, .10), (2.4, 1.0), (1.5, .6), (4.0, 2.5), (3.5, 2.5), (.20, .07)],
}


def generate_alerts(n_per_class: int = 600) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        p = PROFILE[cls]
        nb = rng.normal(*p[0], n_per_class).clip(.3, .95)
        na = rng.normal(*p[1], n_per_class).clip(.02, .9)
        na = np.minimum(na, nb - .02)
        d = pd.DataFrame({
            "ndvi_before": nb, "ndvi_after": na, "ndvi_drop": nb - na,
            "nbr_drop": rng.normal(*p[2], n_per_class).clip(0, .9),
            "area_ha": np.exp(rng.normal(*p[3], n_per_class)).clip(.1, 800),
            "elongation": rng.normal(*p[4], n_per_class).clip(1, 8),
            "dist_road_km": rng.normal(*p[5], n_per_class).clip(0, 15),
            "dist_river_km": rng.normal(*p[6], n_per_class).clip(0, 15),
            "texture": rng.normal(*p[7], n_per_class).clip(.01, .5),
            "driver": cls})
        rows.append(d)
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def mask_cloud(X: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Simulasi tutupan awan: sebagian fitur SPEKTRAL hilang -> diisi median latih.
    Fitur geometri/lokasi (luas, bentuk, jarak) tetap tersedia."""
    Xc = X.copy()
    spectral = ["ndvi_before", "ndvi_after", "ndvi_drop", "nbr_drop", "texture"]
    hide = rng.random(len(Xc)) < frac
    Xc.loc[hide, spectral] = np.nan
    return Xc


def main():
    df = generate_alerts()
    X, y = df[FEATURES], df["driver"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    rep = classification_report(yte, pred, output_dict=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(yte, pred, labels=CLASSES, xticks_rotation=30,
                                            cmap="Greens", ax=ax, colorbar=False)
    ax.set_title(f"Atribusi penyebab deforestasi (akurasi {acc:.2f}, data simulasi)")
    fig.tight_layout(); fig.savefig(OUT / "01_confusion_matrix.png", dpi=140); plt.close(fig)

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(imp.index, imp.values, color="#1B7F5C"); ax.set_title("Fitur apa yang paling menentukan?")
    ax.set_xlabel("Feature importance (Random Forest)"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "02_feature_importance.png", dpi=140); plt.close(fig)

    # --- ketahanan terhadap awan: latih ulang dengan imputasi median ---
    med = Xtr.median()
    fracs = [0, .1, .2, .3, .5, .7]
    robust = []
    for f in fracs:
        Xte_c = mask_cloud(Xte, f).fillna(med)
        robust.append(f1_score(yte, model.predict(Xte_c), average="macro"))
    # model yang dilatih juga dengan data berawan (augmentasi) -> lebih tahan
    Xaug = pd.concat([Xtr, mask_cloud(Xtr, .4)]).fillna(med)
    yaug = pd.concat([ytr, ytr])
    model_aug = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1).fit(Xaug, yaug)
    robust_aug = [f1_score(yte, model_aug.predict(mask_cloud(Xte, f).fillna(med)), average="macro")
                  for f in fracs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([f * 100 for f in fracs], robust, marker="o", label="Model biasa", color="#C0392B")
    ax.plot([f * 100 for f in fracs], robust_aug, marker="s", label="Model dilatih dengan augmentasi awan",
            color="#1B7F5C")
    ax.set(title="Ketahanan terhadap tutupan awan", xlabel="% alert dengan fitur spektral hilang",
           ylabel="Macro F1", ylim=(0, 1.02)); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "03_ketahanan_awan.png", dpi=140); plt.close(fig)

    # --- keluaran operasional: tabel alert baru + penyebab + tingkat keyakinan ---
    new = generate_alerts(4).head(12).drop(columns="driver")
    proba = model.predict_proba(new[FEATURES])
    out = new.round(2).copy()
    out["penyebab_prediksi"] = model.classes_[proba.argmax(1)]
    out["keyakinan"] = proba.max(1).round(2)
    out["perlu_verifikasi_lapangan"] = out["keyakinan"] < 0.6
    out.to_csv(OUT / "contoh_prediksi_alert_baru.csv", index=False)

    res = {
        "jumlah_alert": int(len(df)), "akurasi_uji": round(float(acc), 3),
        "macro_f1_uji": round(float(rep["macro avg"]["f1-score"]), 3),
        "f1_per_kelas": {c: round(rep[c]["f1-score"], 3) for c in CLASSES},
        "fitur_terpenting": imp.sort_values(ascending=False).round(3).head(5).to_dict(),
        "ketahanan_awan_macroF1": {"persen_awan": [int(f * 100) for f in fracs],
                                   "model_biasa": [round(v, 3) for v in robust],
                                   "model_augmentasi": [round(v, 3) for v in robust_aug]},
        "catatan": "Seluruh angka berasal dari DATA SIMULASI, bukan dari citra satelit nyata.",
    }
    (OUT / "hasil.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
