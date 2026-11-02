"""Add LLM-examiner scores and WavLM audio embeddings; compare with 5-fold CV."""
import json
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

O = "outputs"
f = pd.read_csv(f"{O}/features/features.csv")
f = f.merge(pd.DataFrame(json.load(open(f"{O}/asr/transcripts.json")))[["filename", "split", "text"]], on=["filename", "split"])
f = f.merge(pd.read_csv(f"{O}/llm-judge/llm_scores.csv"), on=["filename", "split"])
f = f.merge(pd.read_csv("Dataset_Final/train.csv"), on="filename", how="left")
emb = {k: np.load(f"{O}/features/emb_{k}.npy") for k in ["mpnet", "bge"]}
widx = pd.read_csv(f"{O}/audio/wavlm_index.csv")
assert (widx.filename.values == f.filename.values).all()
for l in [3, 6, 9, 12]:
    emb[f"wavlm{l}"] = np.load(f"{O}/audio/wavlm_l{l}.npy")

LLM = [c for c in f.columns if c.startswith("llm_")]
HAND = [c for c in f.columns if c not in ["filename", "split", "text", "label"] + LLM]
tr = ((f.split == "train") & (f.label > 0)).values
y = f.label[tr].values
print("corr with score:", f[tr][[c for c in LLM if c.endswith("score")]].corrwith(f.label[tr]).round(3).to_dict())

def rep(name, p):
    print(f"{name:20s} pearson={pearsonr(p, y)[0]:.3f} rmse={np.sqrt(np.mean((np.clip(p, 1, 5) - y) ** 2)):.3f}")

def cv(make, X, reps=2):
    X = X[tr]
    oof = np.zeros((reps, len(y)))
    for r in range(reps):
        for a, b in KFold(5, shuffle=True, random_state=r).split(y):
            oof[r, b] = make().fit(X[a], y[a]).predict(X[b])
    return oof.mean(0)

lin = lambda a: (lambda: make_pipeline(StandardScaler(), Ridge(a)))
gbm = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=20,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
Xh, Xhl = f[HAND].fillna(0).values, f[HAND + LLM].fillna(0).values
runs = {
    "hand_ridge": (lin(10), Xh),
    "hand+llm_ridge": (lin(10), Xhl),
    "hand+llm_lgbm": (gbm, Xhl),
    "llm_only_ridge": (lin(1), f[LLM].values),
    "char_tfidf": (lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True), Ridge(3)), f.text.values),
    "mpnet_ridge": (lambda: Ridge(3), emb["mpnet"]),
    "bge_ridge": (lambda: Ridge(3), emb["bge"]),
}
for l in [3, 6, 9, 12]:
    runs[f"wavlm{l}_ridge"] = (lin(1000), emb[f"wavlm{l}"])
oofs = {}
for n, (m, X) in runs.items():
    oofs[n] = cv(m, X)
    rep(n, oofs[n])

for keys in [["hand_ridge", "char_tfidf", "mpnet_ridge", "bge_ridge"], list(oofs)]:
    P = np.column_stack([oofs[k] for k in keys])
    st = np.zeros(len(y))
    for a, b in KFold(5, shuffle=True, random_state=42).split(y):
        st[b] = Ridge(1, positive=True).fit(P[a], y[a]).predict(P[b])
    rep(f"stack({len(keys)} models)", st)
print("weights:", dict(zip(oofs, Ridge(1, positive=True).fit(np.column_stack(list(oofs.values())), y).coef_.round(2))))
