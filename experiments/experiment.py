"""Compare simple models on each feature group with 5-fold CV (x3 repeats)."""
import json, sys
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

O = "outputs"
feats = pd.read_csv(f"{O}/features/features.csv")
docs = pd.DataFrame(json.load(open(f"{O}/asr/transcripts.json")))[["filename", "split", "text"]]
emb = {k: np.load(f"{O}/features/emb_{k}.npy") for k in ["mpnet", "bge"]}
feats = feats.merge(docs, on=["filename", "split"])  # same row order as embeddings
labels = pd.read_csv("Dataset_Final/train.csv")
feats = feats.merge(labels, on="filename", how="left")
tr = (feats.split == "train") & (feats.label > 0)
y = feats.label[tr].values
HAND = [c for c in feats.columns if c not in ("filename", "split", "text", "label")]

print("correlation of each feature with the score:")
print(feats[tr][HAND].corrwith(feats.label[tr]).sort_values().round(2).to_string())

def cv(make, X, reps=3):
    oof = np.zeros((reps, len(y)))
    for r in range(reps):
        for a, b in KFold(5, shuffle=True, random_state=r).split(y):
            m = make().fit(X[a] if not isinstance(X, pd.Series) else X.iloc[a], y[a])
            oof[r, b] = m.predict(X[b] if not isinstance(X, pd.Series) else X.iloc[b])
    p = oof.mean(0)
    return p

Xh = feats[tr][HAND].fillna(0).values
models = {
    "hand_ridge": (lambda: make_pipeline(StandardScaler(), Ridge(10)), Xh),
    "hand_lgbm": (lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1), Xh),
    "char_tfidf": (lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True), Ridge(3)), feats.text[tr].reset_index(drop=True)),
    "mpnet_ridge": (lambda: Ridge(3), emb["mpnet"][tr.values]),
    "bge_ridge": (lambda: Ridge(3), emb["bge"][tr.values]),
    "hand+bge_ridge": (lambda: make_pipeline(StandardScaler(), Ridge(300)), np.hstack([Xh, emb["bge"][tr.values]])),
}
oofs = {}
for name, (make, X) in models.items():
    p = oofs[name] = cv(make, X)
    print(f"{name:16s} pearson={pearsonr(p, y)[0]:.3f} rmse={np.sqrt(np.mean((np.clip(p,1,5) - y) ** 2)):.3f}")

P = np.column_stack(list(oofs.values()))
avg = P.mean(1)
print(f"{'average_all':16s} pearson={pearsonr(avg, y)[0]:.3f} rmse={np.sqrt(np.mean((np.clip(avg,1,5) - y) ** 2)):.3f}")
st = cv(lambda: Ridge(1, positive=True), P, reps=1)
print(f"{'stacked':16s} pearson={pearsonr(st, y)[0]:.3f} rmse={np.sqrt(np.mean((np.clip(st,1,5) - y) ** 2)):.3f}")
print("stack weights:", dict(zip(oofs, Ridge(1, positive=True).fit(P, y).coef_.round(2))))
