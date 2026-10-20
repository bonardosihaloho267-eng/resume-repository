"""
Studi Kasus 3 - Carrier (Career) and Competition in Industrial Era with GIS
Narasumber: Widi Prima Deva (Citycare Property, New Zealand)

Skenario : Perusahaan manajemen properti mengelola 250 aset bangunan di Kota
           Bengkulu dan memiliki 4 depot teknisi. Manajer ingin tahu:
           (a) aset mana dilayani depot mana (analisis jarak terdekat),
           (b) aset berkondisi buruk yang berada JAUH dari depot (prioritas),
           (c) di mana sebaiknya depot baru dibangun (k-means berbobot).
Catatan  : Koordinat aset SINTETIS di sekitar pusat Kota Bengkulu; lokasi depot
           hanya ilustrasi. Hanya memakai numpy/pandas/matplotlib (tanpa GIS
           library berat) agar mudah dijalankan; hasil juga diekspor ke GeoJSON
           sehingga bisa dibuka di QGIS / geojson.io.

Keterampilan GIS yang dilatih: data spasial (lat/lon), jarak haversine,
analisis proximity/buffer, clustering spasial, visualisasi peta, ekspor GeoJSON.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
R_EARTH_KM = 6371.0088
CENTER = (-3.7928, 102.2608)       # pusat Kota Bengkulu (perkiraan)
SERVICE_RADIUS_KM = 3.0

DEPOTS = pd.DataFrame({
    "depot": ["Depot Utara", "Depot Selatan", "Depot Barat", "Depot Timur"],
    "lat": [-3.74, -3.85, -3.79, -3.80],
    "lon": [102.27, 102.28, 102.22, 102.33],
})


def haversine_km(lat1, lon1, lat2, lon2):
    """Jarak lingkaran besar (km); mendukung broadcasting numpy."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def make_assets(n=250) -> pd.DataFrame:
    # tiga klaster permukiman/komersial + sebaran acak
    centers = np.array([[-3.79, 102.26], [-3.83, 102.30], [-3.76, 102.31]])
    idx = rng.integers(0, len(centers), n)
    lat = centers[idx, 0] + rng.normal(0, 0.020, n)
    lon = centers[idx, 1] + rng.normal(0, 0.022, n)
    cond = rng.beta(4, 2.2, n) * 100                     # skor kondisi 0-100
    return pd.DataFrame({
        "asset_id": [f"A{i:03d}" for i in range(n)], "lat": lat, "lon": lon,
        "condition_score": cond.round(1),
        "type": rng.choice(["Rumah", "Ruko", "Gudang", "Kantor"], n, p=[.45, .25, .15, .15]),
    })


def assign_nearest(assets: pd.DataFrame, depots: pd.DataFrame) -> pd.DataFrame:
    d = haversine_km(assets[["lat"]].to_numpy(), assets[["lon"]].to_numpy(),
                     depots["lat"].to_numpy()[None, :], depots["lon"].to_numpy()[None, :])
    a = assets.copy()
    a["nearest_depot"] = depots["depot"].to_numpy()[d.argmin(axis=1)]
    a["dist_km"] = d.min(axis=1).round(2)
    a["in_service_radius"] = a["dist_km"] <= SERVICE_RADIUS_KM
    return a


def propose_new_depot(assets: pd.DataFrame) -> tuple[float, float]:
    """Lokasi depot baru = pusat massa klaster aset 'terlayani buruk',
    diberi bobot oleh seberapa buruk kondisi aset (1 - skor/100)."""
    poor = assets[~assets["in_service_radius"] | (assets["condition_score"] < 50)]
    w = 1 - poor["condition_score"].to_numpy() / 100
    km = KMeans(n_clusters=1, n_init=10, random_state=0)
    km.fit(poor[["lat", "lon"]].to_numpy(), sample_weight=w)
    return tuple(km.cluster_centers_[0])


def coverage(assets, depots):
    a = assign_nearest(assets, depots)
    return float(a["in_service_radius"].mean() * 100), a


def to_geojson(assets, depots, path):
    feats = []
    for r in assets.itertuples():
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                      "properties": {"id": r.asset_id, "type": r.type,
                                     "condition_score": r.condition_score,
                                     "nearest_depot": r.nearest_depot, "dist_km": r.dist_km,
                                     "priority": bool(r.priority)}})
    for r in depots.itertuples():
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                      "properties": {"depot": r.depot}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False))


def main():
    assets = make_assets()
    cov_before, a = coverage(assets, DEPOTS)
    a["priority"] = (a["condition_score"] < 45) & (~a["in_service_radius"])
    new_lat, new_lon = propose_new_depot(a)
    depots2 = pd.concat([DEPOTS, pd.DataFrame({"depot": ["Depot Baru (usulan)"],
                                               "lat": [new_lat], "lon": [new_lon]})], ignore_index=True)
    cov_after, a2 = coverage(assets, depots2)
    a2["priority"] = (a2["condition_score"] < 45) & (~coverage(assets, DEPOTS)[1]["in_service_radius"])

    summary = {
        "jumlah_aset": len(assets),
        "radius_layanan_km": SERVICE_RADIUS_KM,
        "cakupan_sebelum_persen": round(cov_before, 1),
        "cakupan_sesudah_persen": round(cov_after, 1),
        "aset_prioritas_kondisi_buruk_dan_jauh": int(a["priority"].sum()),
        "usulan_depot_baru": {"lat": round(new_lat, 5), "lon": round(new_lon, 5)},
        "beban_aset_per_depot_sesudah": a2["nearest_depot"].value_counts().to_dict(),
        "jarak_rata2_km": {"sebelum": round(float(a["dist_km"].mean()), 2),
                           "sesudah": round(float(a2["dist_km"].mean()), 2)},
    }
    a2.to_csv(OUT / "aset_terlayani.csv", index=False)
    to_geojson(a2, depots2, OUT / "peta.geojson")
    (OUT / "ringkasan.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    for ax, ass, dep, title in [(axs[0], a, DEPOTS, f"Sebelum (cakupan {cov_before:.0f}%)"),
                                (axs[1], a2, depots2, f"Sesudah depot baru (cakupan {cov_after:.0f}%)")]:
        sc = ax.scatter(ass["lon"], ass["lat"], c=ass["condition_score"], cmap="RdYlGn",
                        s=22, edgecolor="none", vmin=0, vmax=100)
        far = ass[~ass["in_service_radius"]]
        ax.scatter(far["lon"], far["lat"], s=60, facecolors="none", edgecolors="k", lw=.7,
                   label="di luar radius 3 km")
        for r in dep.itertuples():
            new = "usulan" in r.depot
            ax.scatter(r.lon, r.lat, marker="*" if not new else "P", s=260,
                       c="#1F3A5F" if not new else "#8E44AD", edgecolor="w", zorder=5)
            ax.add_patch(plt.Circle((r.lon, r.lat), SERVICE_RADIUS_KM / 111.32 / np.cos(np.radians(CENTER[0])),
                                    fill=False, ls="--", lw=.8, color="#1F3A5F"))
        ax.set(title=title, xlabel="Bujur"); ax.set_aspect("equal"); ax.grid(alpha=.25)
    axs[0].set_ylabel("Lintang"); axs[0].legend(loc="lower left", fontsize=8)
    fig.colorbar(sc, ax=axs, label="Skor kondisi aset (0=buruk, 100=baik)", shrink=.85)
    fig.suptitle("Analisis proximity aset properti dan usulan depot baru (data sintetis)")
    fig.savefig(OUT / "peta_cakupan.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
