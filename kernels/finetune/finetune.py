# Step 5: fine-tune WavLM-large to predict the grammar score directly from audio.
# 5-fold cross-validation: for each fold we train on 4/5 of the training clips and predict
# the held-out 1/5 (out-of-fold predictions, used to measure and stack) and the test set.

import glob, os, random
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from transformers import WavLMModel

SR, CROP, EPOCHS, BATCH, LR = 16000, 15, 8, 4, 3e-5
N_LAYERS = 21  # layers 18-21 gave the best frozen embeddings; drop the top 3 to save compute
torch.manual_seed(0); np.random.seed(0); random.seed(0)

root = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
train = pd.read_csv(f"{root}/train.csv")
train = train[train.label > 0].reset_index(drop=True)  # drop the score-0 batch, as in the main pipeline
test = pd.read_csv(f"{root}/test.csv")


def load(path):
    a, _ = librosa.load(path, sr=SR, mono=True)
    a = a[: SR * 60] if len(a) else np.zeros(SR, dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-7)


tr_audio = [load(f"{root}/train/{f}") for f in train.filename]
te_audio = [load(f"{root}/test/{f}") for f in test.filename]
y = train.label.values.astype(np.float32)
print("loaded audio")


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm.encoder.layers = self.wavlm.encoder.layers[:N_LAYERS]
        self.wavlm.feature_extractor._freeze_parameters()  # keep the low-level CNN fixed
        self.wavlm.gradient_checkpointing_enable()  # recompute activations in backward to fit in T4 memory
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(1024, 1))

    def forward(self, x):
        h = self.wavlm(x).last_hidden_state
        return self.head(h.mean(1)).squeeze(-1) + 3.5  # start predictions near the average score


def crop(a, sec):
    n = SR * sec
    if len(a) <= n:
        return np.pad(a, (0, n - len(a)))
    s = random.randint(0, len(a) - n)
    return a[s: s + n]


@torch.no_grad()
def predict(model, audios):
    """Average the prediction over consecutive 15 s windows covering the whole clip."""
    model.eval()
    out = []
    for a in audios:
        n = SR * CROP
        wins = [a[s: s + n] for s in range(0, max(len(a) - n, 0) + 1, n // 2)] or [a]
        wins = [np.pad(w, (0, n - len(w))) for w in wins]
        x = torch.tensor(np.stack(wins)).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x).float().mean().item())
    return np.array(out)


oof = np.zeros(len(y))
test_pred = np.zeros(len(test))
for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(y)):
    model = Scorer().cuda()
    opt = torch.optim.AdamW([{"params": model.wavlm.parameters(), "lr": LR},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    steps = EPOCHS * (len(a) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(a)
        for i in range(0, len(idx) - BATCH + 1, BATCH):
            bi = idx[i: i + BATCH]
            x = torch.tensor(np.stack([crop(tr_audio[j], CROP) for j in bi])).cuda()
            t = torch.tensor(y[bi]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = nn.functional.mse_loss(model(x).float(), t)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        p = predict(model, [tr_audio[j] for j in b])
        print(f"fold {fold} epoch {ep} val rmse={np.sqrt(np.mean((np.clip(p, 1, 5) - y[b]) ** 2)):.3f} "
              f"pearson={np.corrcoef(p, y[b])[0, 1]:.3f}", flush=True)
    oof[b] = p
    test_pred += predict(model, te_audio) / 5
    del model, opt
    torch.cuda.empty_cache()

print(f"OOF rmse={np.sqrt(np.mean((np.clip(oof, 1, 5) - y) ** 2)):.3f} pearson={np.corrcoef(oof, y)[0, 1]:.3f}")
pd.DataFrame({"filename": train.filename, "pred": oof}).to_csv("/kaggle/working/ft_oof.csv", index=False)
pd.DataFrame({"filename": test.filename, "pred": test_pred}).to_csv("/kaggle/working/ft_test.csv", index=False)
