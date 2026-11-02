# Step 4: audio embeddings.
# A pretrained speech model (WavLM) "listens" to the raw audio. We average its hidden
# states over time to get one vector per recording per layer; middle layers capture
# fluency and pronunciation, which the text-only features cannot see.

import glob, os
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, WavLMModel

root = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
jobs = [("train", f) for f in pd.read_csv(f"{root}/train.csv").filename] + \
       [("test", f) for f in pd.read_csv(f"{root}/test.csv").filename]

fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to("cuda").eval()
LAYERS = [3, 6, 9, 12]

embs = {l: [] for l in LAYERS}
for i, (split, fname) in enumerate(jobs):
    audio, _ = librosa.load(f"{root}/{split}/{fname}", sr=16000, mono=True)
    audio = audio[: 16000 * 60] if len(audio) else np.zeros(16000, dtype=np.float32)
    inp = fe(audio, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
    with torch.no_grad():
        hs = model(inp, output_hidden_states=True).hidden_states
    for l in LAYERS:
        embs[l].append(hs[l][0].mean(0).cpu().numpy())
    if i % 100 == 0:
        print(i, "/", len(jobs))

for l in LAYERS:
    np.save(f"/kaggle/working/wavlm_l{l}.npy", np.array(embs[l]))
pd.DataFrame(jobs, columns=["split", "filename"]).to_csv("/kaggle/working/wavlm_index.csv", index=False)
print("done")
