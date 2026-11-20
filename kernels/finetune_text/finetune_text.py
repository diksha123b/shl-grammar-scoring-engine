# Step 6: fine-tune DeBERTa-v3-large to predict the grammar score from the Whisper transcript.
# Grammar is mostly a property of the words, and so far the transcripts only reach the model
# through fixed features and frozen embeddings. Here a text model is trained on the scores directly.
# Same speaker-grouped folds as src/pipeline.py, so the out-of-fold predictions are honest and
# can be stacked. Small data -> results are averaged over 3 seeds.

import glob, json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MODEL, MAX_LEN, EPOCHS, BATCH, LR, SEEDS = "microsoft/deberta-v3-large", 256, 6, 8, 1e-5, (0, 1, 2)

root = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
train = pd.read_csv(f"{root}/train.csv")
train = train[train.label > 0].reset_index(drop=True)  # drop the score-0 batch, as in the main pipeline
test = pd.read_csv(f"{root}/test.csv")
docs = pd.DataFrame(json.load(open(glob.glob("/kaggle/input/**/transcripts.json", recursive=True)[0])))
text = docs.set_index(["split", "filename"]).text  # names repeat across train/test
tr_text = [text["train", f].strip() for f in train.filename]
te_text = [text["test", f].strip() for f in test.filename]
y = train.label.values.astype(np.float32)

# Voice groups: same recipe as src/pipeline.py (WavLM-large layer 6 embeddings from the audio2 kernel)
idx = pd.read_csv(glob.glob("/kaggle/input/**/index.csv", recursive=True)[0])
V = np.load(glob.glob("/kaggle/input/**/wavlmL_l6.npy", recursive=True)[0])
pos = idx[idx.split == "train"].reset_index().set_index("filename").loc[train.filename, "index"].values
V = normalize(V[pos] - V[pos].mean(0))
groups = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                 distance_threshold=0.25).fit_predict(V)
print(len(set(groups)), "voice groups")

tok = AutoTokenizer.from_pretrained(MODEL)


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32)  # fp32 weights; autocast handles fp16
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.config.hidden_size, 1))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)  # mean over real tokens
        return self.head(pooled).squeeze(-1) + 3.5  # start predictions near the average score


def encode(texts):
    t = tok(list(texts), max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt")
    return t.input_ids.cuda(), t.attention_mask.cuda()


@torch.no_grad()
def predict(model, texts):
    model.eval()
    out = []
    for i in range(0, len(texts), 32):
        ids, mask = encode(texts[i: i + 32])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(ids, mask).float().cpu().numpy())
    return np.concatenate(out)


oof = np.zeros(len(y))
test_pred = np.zeros(len(test))
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    oof_s = np.zeros(len(y))
    for fold, (a, b) in enumerate(GroupKFold(5).split(V, y, groups)):
        model = Scorer().cuda()
        opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": LR},
                                 {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
        steps = EPOCHS * (len(a) // BATCH)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.cuda.amp.GradScaler()
        for ep in range(EPOCHS):
            model.train()
            order = np.random.permutation(a)
            for i in range(0, len(order) - BATCH + 1, BATCH):
                bi = order[i: i + BATCH]
                ids, mask = encode([tr_text[j] for j in bi])
                t = torch.tensor(y[bi]).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = nn.functional.mse_loss(model(ids, mask).float(), t)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            p = predict(model, [tr_text[j] for j in b])
            print(f"seed {seed} fold {fold} epoch {ep} val rmse={np.sqrt(np.mean((np.clip(p, 1, 5) - y[b]) ** 2)):.3f} "
                  f"pearson={np.corrcoef(p, y[b])[0, 1]:.3f}", flush=True)
        oof_s[b] = p
        test_pred += predict(model, te_text) / (5 * len(SEEDS))
        del model, opt
        torch.cuda.empty_cache()
    print(f"seed {seed} OOF rmse={np.sqrt(np.mean((np.clip(oof_s, 1, 5) - y) ** 2)):.3f} "
          f"pearson={np.corrcoef(oof_s, y)[0, 1]:.3f}", flush=True)
    oof += oof_s / len(SEEDS)

print(f"OOF (seed avg) rmse={np.sqrt(np.mean((np.clip(oof, 1, 5) - y) ** 2)):.3f} pearson={np.corrcoef(oof, y)[0, 1]:.3f}")
pd.DataFrame({"filename": train.filename, "pred": oof}).to_csv("/kaggle/working/fttx_oof.csv", index=False)
pd.DataFrame({"filename": test.filename, "pred": test_pred}).to_csv("/kaggle/working/fttx_test.csv", index=False)
