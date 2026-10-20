"""
Studi Kasus 4 - Introduction to Machine Learning
Narasumber: Dr. Taniza Tajuddin (UiTM Cawangan Kedah)

Skenario : Sebuah laboratorium ingin mengklasifikasikan jenis anggur (3 kultivar)
           dari 13 hasil uji kimia. Dataset "wine" sudah tersedia di
           scikit-learn (tanpa unduhan).

Yang didemonstrasikan (sesuai materi pengantar ML):
  1. Alur kerja ML  : split data -> pra-pemrosesan -> latih -> evaluasi
  2. Supervised     : membandingkan 4 algoritma klasifikasi (cross-validation)
  3. Overfitting    : kurva kedalaman Decision Tree (underfit vs overfit)
  4. Unsupervised   : k-means tanpa label, lalu dicek dengan Adjusted Rand Index
  5. Evaluasi       : confusion matrix, precision/recall/F1
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, adjusted_rand_score,
                             classification_report, accuracy_score)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
SEED = 42


def main():
    data = load_wine()
    X, y = data.data, data.target
    # --- 1. split: data uji TIDAK boleh disentuh saat memilih model ---
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    # --- 2. supervised: bandingkan algoritma dengan cross-validation ---
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_res = {}
    for name, m in models.items():
        s = cross_val_score(m, Xtr, ytr, cv=cv)
        cv_res[name] = {"cv_mean": round(s.mean(), 3), "cv_std": round(s.std(), 3)}
    best_name = max(cv_res, key=lambda k: cv_res[k]["cv_mean"])
    best = models[best_name].fit(Xtr, ytr)
    pred = best.predict(Xte)
    test_acc = accuracy_score(yte, pred)
    report = classification_report(yte, pred, target_names=data.target_names, output_dict=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=data.target_names,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {best_name}\n(akurasi data uji {test_acc:.3f})")
    fig.tight_layout(); fig.savefig(OUT / "01_confusion_matrix.png", dpi=140); plt.close(fig)

    # --- 3. overfitting vs underfitting ---
    depths = range(1, 13)
    tr, va = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, ytr)
        tr.append(m.score(Xtr, ytr))
        va.append(cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=SEED),
                                  Xtr, ytr, cv=cv).mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, tr, marker="o", label="Akurasi data latih", color="#C0392B")
    ax.plot(depths, va, marker="s", label="Akurasi validasi (CV)", color="#1F3A5F")
    ax.set(title="Kompleksitas model vs generalisasi (Decision Tree)",
           xlabel="Kedalaman maksimum pohon", ylabel="Akurasi", ylim=(0.5, 1.02))
    ax.annotate("underfitting", (1, va[0]), xytext=(1.3, 0.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("jarak latih-validasi\nmelebar = overfitting", (9, va[8]), xytext=(6.5, 0.68),
                arrowprops=dict(arrowstyle="->"))
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "02_overfitting.png", dpi=140); plt.close(fig)

    # --- 4. unsupervised: label TIDAK dipakai saat clustering ---
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Xs)
    ari = adjusted_rand_score(y, km.labels_)
    # proyeksi 2D dengan PCA sederhana (SVD) untuk visualisasi
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    P = Xs @ Vt[:2].T
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    ax[0].scatter(P[:, 0], P[:, 1], c=y, cmap="viridis", s=18); ax[0].set_title("Label asli (kultivar)")
    ax[1].scatter(P[:, 0], P[:, 1], c=km.labels_, cmap="viridis", s=18)
    ax[1].set_title(f"Hasil k-means tanpa label (ARI={ari:.2f})")
    for a in ax: a.set(xlabel="PC1", ylabel="PC2"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "03_kmeans.png", dpi=140); plt.close(fig)

    result = {
        "ukuran_data": {"total": int(len(y)), "latih": int(len(ytr)), "uji": int(len(yte)),
                        "jumlah_fitur": int(X.shape[1])},
        "cross_validation_data_latih": cv_res,
        "model_terpilih": best_name,
        "akurasi_data_uji": round(float(test_acc), 3),
        "laporan_klasifikasi_f1": {k: round(v["f1-score"], 3) for k, v in report.items()
                                   if k in data.target_names},
        "decision_tree_overfitting": {"akurasi_latih_depth12": round(tr[-1], 3),
                                      "akurasi_validasi_depth12": round(va[-1], 3),
                                      "validasi_terbaik": round(max(va), 3),
                                      "depth_terbaik": int(list(depths)[int(np.argmax(va))])},
        "kmeans_ARI": round(float(ari), 3),
    }
    (OUT / "hasil.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
