"""Final modelling pipeline (submission v11): features -> base models -> stacked grammar score.

The heavy steps (speech-to-text, grammar/LLM features, audio embeddings, fine-tuning) run as
Kaggle GPU scripts in `kernels/`; their outputs are downloaded into `outputs/`. This module
loads those outputs, trains the base models with speaker-grouped cross-validation, and stacks them.
"""
import glob
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

OUT = "outputs"
DATA = "Dataset_Final"
TEXT_EMB = ["mpnet", "bge"]
# Audio embeddings computed on the first 45 s of every clip (test clips are ~45 s, train ~60 s)
AUDIO_EMB = [f"wavlmL_l{l}" for l in (6, 9, 12, 15, 18, 21, 24)] + [f"whisper_l{l}" for l in (8, 16, 20, 24, 28, 32)]
FINETUNED = {"whisper_finetuned": ("finetune-whisper-g", "ftwg"),  # trained with the same grouped folds
             "text_finetuned": ("finetune-text", "fttx")}


def load_table(out=OUT, data=DATA):
    """One row per clip (train + test) with hand features, transcript, LLM scores and label."""
    f = pd.read_csv(f"{out}/features/features.csv")
    # Test clips are shorter than training clips: replace counts that grow with clip length by
    # per-second rates so that train and test features are comparable
    f["sentences_per_sec"] = f.n_sentences / f.duration
    f["long_pauses_per_sec"] = f.n_long_pauses / f.duration
    f = f.drop(columns=["duration", "n_words", "n_sentences", "n_long_pauses"])
    docs = pd.DataFrame(json.load(open(f"{out}/asr/transcripts.json")))[["filename", "split", "text"]]
    f = f.merge(docs, on=["filename", "split"])
    f = f.merge(pd.read_csv(f"{out}/llm-judge/llm_scores.csv"), on=["filename", "split"])
    # File names repeat between the train and test folders, so attach labels to training rows only
    labels = pd.read_csv(f"{data}/train.csv").assign(split="train")
    return f.merge(labels, on=["filename", "split"], how="left")


def load_embeddings(f, out=OUT):
    """Text and audio embedding matrices, row-aligned with `f`."""
    emb = {k: np.load(f"{out}/features/emb_{k}.npy") for k in TEXT_EMB}
    idx = pd.read_csv(f"{out}/audio45/index.csv")
    assert (idx.filename.values == f.filename.values).all() and (idx.split.values == f.split.values).all()
    for k in AUDIO_EMB:
        emb[k] = np.load(f"{out}/audio45/{k}.npy")
    return emb


def column_groups(f):
    llm = [c for c in f.columns if c.startswith("llm_")]
    hand = [c for c in f.columns if c not in ["filename", "split", "text", "label"] + llm]
    return hand, llm


def voice_folds(f, train_mask, out=OUT, n_folds=5):
    """Speaker-grouped CV folds.

    Many training clips have a near-identical voice in another training clip (likely the same
    speaker), while test clips never do. Plain KFold lets models "recognise the voice" and
    overstates their quality, so clips are clustered by voice similarity (WavLM-large layer 6,
    which mostly carries speaker/acoustic information) and each cluster stays inside one fold.
    """
    V = np.load(f"{out}/audio2/wavlmL_l6.npy")[train_mask]
    V = normalize(V - V.mean(0))
    groups = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=0.25).fit_predict(V)
    return groups, list(GroupKFold(n_folds).split(V, groups=groups))


def base_models(f, emb):
    """name -> (model factory, feature matrix for all rows)."""
    hand, llm = column_groups(f)
    ridge = lambda a: (lambda: make_pipeline(StandardScaler(), Ridge(a)))
    gbm = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=20,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    tfidf = lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True),
                                  Ridge(3))
    Xh, Xhl = f[hand].fillna(0).values, f[hand + llm].fillna(0).values
    models = {
        "hand_ridge": (ridge(10), Xh),
        "hand+llm_ridge": (ridge(10), Xhl),
        "hand+llm_lgbm": (gbm, Xhl),
        "llm_only_ridge": (ridge(1), f[llm].values),
        "char_tfidf": (tfidf, f.text.values),
        "mpnet_ridge": (lambda: Ridge(3), emb["mpnet"]),
        "bge_ridge": (lambda: Ridge(3), emb["bge"]),
    }
    for k in AUDIO_EMB:
        models[f"{k}_ridge"] = (ridge(1000), emb[k])
    return models


def metrics(pred, y):
    """Pearson correlation and RMSE (predictions clipped to the 1-5 score range)."""
    return pearsonr(pred, y)[0], float(np.sqrt(np.mean((np.clip(pred, 1, 5) - y) ** 2)))


def run(out=OUT, data=DATA, verbose=True):
    """Train everything; returns a dict with the data, per-model OOF/test predictions and the stack."""
    f = load_table(out, data)
    emb = load_embeddings(f, out)
    tr = ((f.split == "train") & (f.label > 0)).values  # drop the score-0 batch (not in rubric, not in test)
    te = (f.split == "test").values
    y = f.label[tr].values
    groups, folds = voice_folds(f, tr, out)

    # Out-of-fold predictions (honest scores + stacking inputs) and full-data test predictions
    # (+ in-sample predictions on the training clips, for the training RMSE)
    oof, test, fit = {}, {}, {}
    for name, (make, X) in base_models(f, emb).items():
        Xtr = X[tr]
        o = np.zeros(len(y))
        for a, b in folds:
            o[b] = make().fit(Xtr[a], y[a]).predict(Xtr[b])
        full = make().fit(Xtr, y)
        oof[name], test[name], fit[name] = o, full.predict(X[te]), full.predict(Xtr)
        if verbose:
            print(f"{name:20s} pearson={metrics(o, y)[0]:.3f} rmse={metrics(o, y)[1]:.3f}")

    # Fine-tuned model: trained on Kaggle with the same folds, so its saved OOF predictions are honest
    for name, (folder, prefix) in FINETUNED.items():
        o = pd.read_csv(f"{out}/{folder}/{prefix}_oof.csv")
        assert (o.filename.values == f.filename[tr].values).all()
        oof[name] = fit[name] = o.pred.values  # in-sample predictions were not saved on Kaggle; OOF stands in
        test[name] = f[te][["filename"]].merge(pd.read_csv(f"{out}/{folder}/{prefix}_test.csv"), on="filename").pred.values
        if verbose:
            print(f"{name:20s} pearson={metrics(oof[name], y)[0]:.3f} rmse={metrics(oof[name], y)[1]:.3f}")

    # Stack: non-negative linear blend of the base models, evaluated with the same grouped folds
    P, T = np.column_stack(list(oof.values())), np.column_stack(list(test.values()))
    stack_oof = np.zeros(len(y))
    for a, b in folds:
        stack_oof[b] = Ridge(1, positive=True).fit(P[a], y[a]).predict(P[b])
    stacker = Ridge(1, positive=True).fit(P, y)
    train_pred = np.clip(stacker.predict(np.column_stack(list(fit.values()))), 1, 5)
    if verbose:
        print(f"{'STACK':20s} pearson={metrics(stack_oof, y)[0]:.3f} rmse={metrics(stack_oof, y)[1]:.3f}")

    sub = f[te][["filename"]].assign(label=np.clip(stacker.predict(T), 1, 5))
    sub = pd.read_csv(f"{data}/test.csv")[["filename"]].merge(sub, on="filename", how="left")
    assert sub.label.notna().all() and len(sub) == 216
    return dict(f=f, emb=emb, tr=tr, te=te, y=y, groups=groups, folds=folds, oof=oof, test=test,
                stack_oof=stack_oof, stacker=stacker, P=P, train_pred=train_pred, submission=sub)
