"""Try different regressors on the best frozen audio embeddings (5-fold CV)."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

f = pd.read_csv("outputs/audio2/index.csv").merge(pd.read_csv("Dataset_Final/train.csv"), on="filename", how="left")
tr = ((f.split == "train") & (f.label > 0)).values
y = f.label[tr].values
E = {k: np.load(f"outputs/audio2/{k}.npy")[tr] for k in ["wavlmL_l18", "wavlmL_l21", "whisper_l32"]}
E["wavlmL_18+21+whisper32"] = np.hstack(list(E.values()))

heads = {
    "ridge a=1000": lambda: make_pipeline(StandardScaler(), Ridge(1000)),
    "ridge a=300": lambda: make_pipeline(StandardScaler(), Ridge(300)),
    "ridge a=3000": lambda: make_pipeline(StandardScaler(), Ridge(3000)),
    "svr rbf C=3": lambda: make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.1)),
    "svr rbf C=10": lambda: make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1)),
    "pca128+svr C=3": lambda: make_pipeline(StandardScaler(), PCA(128, random_state=0), SVR(C=3)),
    "kernel ridge rbf": lambda: make_pipeline(StandardScaler(), KernelRidge(alpha=1, kernel="rbf", gamma=1 / 2000)),
}
for en, X in E.items():
    for hn, make in heads.items():
        if hn == "kernel ridge rbf":
            make = (lambda g: (lambda: make_pipeline(StandardScaler(), KernelRidge(alpha=1, kernel="rbf", gamma=g))))(0.5 / X.shape[1])
        p = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=0).split(y):
            m = make().fit(X[a], y[a])
            p[b] = m.predict(X[b]) + (y[a].mean() if hn == "kernel ridge rbf" and False else 0)
        print(f"{en:24s} {hn:18s} pearson={pearsonr(p, y)[0]:.3f} rmse={np.sqrt(np.mean((np.clip(p, 1, 5) - y) ** 2)):.3f}", flush=True)
