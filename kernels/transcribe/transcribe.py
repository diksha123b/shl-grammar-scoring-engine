# Step 1: speech -> text.
# Transcribes every train and test recording with Whisper (large-v3) on a GPU and
# saves the text plus word timings, which later steps turn into grammar features.

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)

import glob, json, os, time
import librosa
import pandas as pd
from faster_whisper import WhisperModel

# Find the competition folder wherever Kaggle mounted it
root = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
print("data folder:", root)

jobs = [("train", f) for f in pd.read_csv(f"{root}/train.csv").filename] + \
       [("test", f) for f in pd.read_csv(f"{root}/test.csv").filename]
print(len(jobs), "files to transcribe")

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Whisper tends to "clean up" speech (drops ums, fixes small mistakes). For grammar
# scoring we want what was actually said, so we prompt it with a verbatim-style example.
VERBATIM_PROMPT = "Umm, so, uh, I think that, like, he go to the school and, hmm, we was there."

# Resume: reuse transcripts from an earlier run if one is attached as input
previous = glob.glob("/kaggle/input/**/transcripts.json", recursive=True)
results = json.load(open(previous[0])) if previous else []
done = {(r["split"], r["filename"]) for r in results}
print(len(done), "already transcribed")

start = time.time()
for i, (split, fname) in enumerate(jobs):
    if (split, fname) in done:
        continue
    # Load audio ourselves (16 kHz mono, what Whisper expects); faster-whisper's own
    # loader clashes with the PyAV version on Kaggle
    audio, _ = librosa.load(f"{root}/{split}/{fname}", sr=16000, mono=True)
    options = dict(language="en", beam_size=5, condition_on_previous_text=False,
                   initial_prompt=VERBATIM_PROMPT)
    try:
        segments, info = model.transcribe(audio, word_timestamps=True, **options)
        segs = list(segments)
    except IndexError:
        # faster-whisper's word-timing step can crash on near-silent clips; keep the text without timings
        print("word timings failed, retrying without them:", fname)
        segments, info = model.transcribe(audio, word_timestamps=False, **options)
        segs = list(segments)
    results.append({
        "split": split,
        "filename": fname,
        "duration": info.duration,
        "text": " ".join(s.text.strip() for s in segs),
        "segments": [{"start": s.start, "end": s.end, "text": s.text,
                      "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob}
                     for s in segs],
        "words": [{"w": w.word, "start": w.start, "end": w.end, "p": w.probability}
                  for s in segs for w in (s.words or [])],
    })
    if i % 50 == 0:
        print(f"{i}/{len(jobs)} done, {time.time() - start:.0f}s elapsed | {fname}: {results[-1]['text'][:100]}")
        json.dump(results, open("/kaggle/working/transcripts.json", "w"))

json.dump(results, open("/kaggle/working/transcripts.json", "w"))
print("finished in", round(time.time() - start), "seconds")
