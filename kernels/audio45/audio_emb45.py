# Step 4c: same as audio_emb2.py, but every clip is cut to its first 45 s.
# Test clips are mostly ~45 s while training clips are ~60 s; cutting both to 45 s
# makes train and test embeddings comparable.
#
# (original header:) Step 4b: larger audio models.
# Same idea as audio_emb.py (average hidden states over time, one vector per layer),
# with bigger speech models: WavLM-large and the encoder of Whisper large-v3.

import glob, os
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, WavLMModel, WhisperFeatureExtractor, WhisperModel

root = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
jobs = [("train", f) for f in pd.read_csv(f"{root}/train.csv").filename] + \
       [("test", f) for f in pd.read_csv(f"{root}/test.csv").filename]
audios = []
for split, fname in jobs:
    a, _ = librosa.load(f"{root}/{split}/{fname}", sr=16000, mono=True)
    audios.append(a[: 16000 * 45] if len(a) else np.zeros(16000, dtype=np.float32))
pd.DataFrame(jobs, columns=["split", "filename"]).to_csv("/kaggle/working/index.csv", index=False)

# WavLM-large (24 layers)
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
m = WavLMModel.from_pretrained("microsoft/wavlm-large", torch_dtype=torch.float16).to("cuda").eval()
LAYERS = [6, 9, 12, 15, 18, 21, 24]
embs = {l: [] for l in LAYERS}
for a in audios:
    x = fe(a, sampling_rate=16000, return_tensors="pt").input_values.to("cuda", torch.float16)
    with torch.no_grad():
        hs = m(x, output_hidden_states=True).hidden_states
    for l in LAYERS:
        embs[l].append(hs[l][0].float().mean(0).cpu().numpy())
for l in LAYERS:
    np.save(f"/kaggle/working/wavlmL_l{l}.npy", np.array(embs[l]))
del m; torch.cuda.empty_cache()
print("wavlm-large done")

# Whisper large-v3 encoder (32 layers); only average over frames that contain audio
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
enc = WhisperModel.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).encoder.to("cuda").eval()
LAYERS = [8, 16, 20, 24, 28, 32]
embs = {l: [] for l in LAYERS}
for a in audios:
    # Whisper's encoder takes 30 s windows: run each 30 s half separately
    vecs = {l: [] for l in LAYERS}
    for start in range(0, min(len(a), 16000 * 45), 16000 * 30):
        chunk = a[start: start + 16000 * 30]
        x = fe(chunk, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        k = max(1, min(1500, int(len(chunk) / 16000 * 50)))
        with torch.no_grad():
            hs = enc(x, output_hidden_states=True).hidden_states
        for l in LAYERS:
            vecs[l].append(hs[l][0, :k].float().mean(0).cpu().numpy())
    for l in LAYERS:
        embs[l].append(np.mean(vecs[l], 0))
for l in LAYERS:
    np.save(f"/kaggle/working/whisper_l{l}.npy", np.array(embs[l]))
print("whisper done")
