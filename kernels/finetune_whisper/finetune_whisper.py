# Step 5b: fine-tune the Whisper large-v3 encoder to predict the grammar score.
# Same 5 folds as finetune.py. To fit the free T4 GPU, the bottom 24 of 32 encoder layers
# stay frozen (run without gradients); only the top 8 layers and a small head are trained.

import glob, os, random
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from transformers import WhisperFeatureExtractor, WhisperModel

SR, CROP, EPOCHS, BATCH, LR, N_FROZEN = 16000, 30, 6, 4, 2e-5, 24
torch.manual_seed(0); np.random.seed(0); random.seed(0)

root = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
train = pd.read_csv(f"{root}/train.csv")
train = train[train.label > 0].reset_index(drop=True)  # drop the score-0 batch, as in the main pipeline
test = pd.read_csv(f"{root}/test.csv")
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")


def load(path):
    a, _ = librosa.load(path, sr=SR, mono=True)
    return a[: SR * 60] if len(a) else np.zeros(SR, dtype=np.float32)


tr_audio = [load(f"{root}/train/{f}") for f in train.filename]
te_audio = [load(f"{root}/test/{f}") for f in test.filename]
y = train.label.values.astype(np.float32)
print("loaded audio")


def run_layer(layer, h):
    out = layer(h, attention_mask=None)
    return out[0] if isinstance(out, tuple) else out  # older transformers versions return a tuple


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = WhisperModel.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float32).encoder
        for p in self.enc.parameters():
            p.requires_grad = False
        for layer in self.enc.layers[N_FROZEN:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.enc.layer_norm.parameters():
            p.requires_grad = True
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(1280, 1))

    def forward(self, feats, n_frames):
        e = self.enc
        with torch.no_grad():  # frozen bottom: same steps as WhisperEncoder.forward
            h = nn.functional.gelu(e.conv1(feats))
            h = nn.functional.gelu(e.conv2(h)).permute(0, 2, 1)
            h = h + e.embed_positions.weight[: h.shape[1]]
            for layer in e.layers[:N_FROZEN]:
                h = run_layer(layer, h)
        for layer in e.layers[N_FROZEN:]:
            h = run_layer(layer, h)
        h = e.layer_norm(h)
        # average only over frames that contain audio (input is padded to 30 s)
        mask = (torch.arange(h.shape[1], device=h.device)[None] < n_frames[:, None]).float()
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.head(pooled).squeeze(-1) + 3.5


def features(chunks):
    x = fe([c for c in chunks], sampling_rate=SR, return_tensors="pt").input_features
    n = torch.tensor([max(1, min(1500, int(len(c) / SR * 50))) for c in chunks])
    return x.cuda(), n.cuda()


def crop(a):
    n = SR * CROP
    if len(a) <= n:
        return a
    s = random.randint(0, len(a) - n)
    return a[s: s + n]


@torch.no_grad()
def predict(model, audios):
    """Average the prediction over the 30 s halves of each clip."""
    model.eval()
    out = []
    for a in audios:
        n = SR * CROP
        chunks = [a[s: s + n] for s in range(0, len(a), n) if len(a[s: s + n]) > SR * 5] or [a]
        x, k = features(chunks)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x, k).float().mean().item())
    return np.array(out)


oof = np.zeros(len(y))
test_pred = np.zeros(len(test))
for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(y)):
    model = Scorer().cuda()
    params = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": LR}, {"params": model.head.parameters(), "lr": 1e-3}],
                            weight_decay=0.01)
    steps = EPOCHS * (len(a) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(a)
        for i in range(0, len(idx) - BATCH + 1, BATCH):
            bi = idx[i: i + BATCH]
            x, k = features([crop(tr_audio[j]) for j in bi])
            t = torch.tensor(y[bi]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = nn.functional.mse_loss(model(x, k).float(), t)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
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
pd.DataFrame({"filename": train.filename, "pred": oof}).to_csv("/kaggle/working/ftw_oof.csv", index=False)
pd.DataFrame({"filename": test.filename, "pred": test_pred}).to_csv("/kaggle/working/ftw_test.csv", index=False)
