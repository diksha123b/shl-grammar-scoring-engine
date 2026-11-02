# Step 2: transcripts -> features.
# Turns each Whisper transcript into numbers that describe grammar and fluency:
#   - grammar errors found by a grammar-correction model (CoEdIT)
#   - how "surprising" the sentences are to a language model (GPT-2 loss)
#   - fluency from word timings (speaking rate, pauses, filler words)
#   - sentence embeddings of the whole transcript (meaning/style vectors)

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)

import difflib, glob, json, re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

device = "cuda"
OUT = "/kaggle/working"
docs = json.load(open(glob.glob("/kaggle/input/**/transcripts.json", recursive=True)[0]))
print(len(docs), "transcripts")

FILLERS = {"um", "umm", "uh", "uhh", "hmm", "er", "erm", "ah", "mm"}


def words_of(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def split_sentences(text, max_words=40):
    """Split on sentence punctuation, then cut very long run-on sentences into chunks."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        w = s.split()
        for i in range(0, len(w), max_words):
            if w[i:i + max_words]:
                out.append(" ".join(w[i:i + max_words]))
    return out


# ---------- 1. fluency + basic text statistics ----------
rows = []
for d in docs:
    toks = words_of(d["text"])
    sents = split_sentences(d["text"])
    ws = d["words"]
    gaps = [b["start"] - a["end"] for a, b in zip(ws, ws[1:])]
    speech_time = (ws[-1]["end"] - ws[0]["start"]) if ws else 0.0
    sent_lens = [len(s.split()) for s in sents] or [0]
    rows.append({
        "filename": d["filename"], "split": d["split"], "duration": d["duration"],
        "n_words": len(toks),
        "n_sentences": len(sents),
        "mean_sent_len": np.mean(sent_lens),
        "std_sent_len": np.std(sent_lens),
        "type_token_ratio": len(set(toks)) / max(len(toks), 1),
        "mean_word_len": np.mean([len(t) for t in toks]) if toks else 0,
        "filler_rate": sum(t in FILLERS for t in toks) / max(len(toks), 1),
        "speech_rate": len(ws) / max(speech_time, 1e-6),
        "speech_ratio": speech_time / max(d["duration"], 1e-6),
        "n_long_pauses": sum(g > 1.0 for g in gaps),
        "mean_pause": np.mean(gaps) if gaps else 0,
        "mean_word_prob": np.mean([w["p"] for w in ws]) if ws else 0,
        "low_prob_word_frac": np.mean([w["p"] < 0.5 for w in ws]) if ws else 0,
        "mean_seg_logprob": np.mean([s["avg_logprob"] for s in d["segments"]]) if d["segments"] else -5,
        "max_no_speech_prob": max([s["no_speech_prob"] for s in d["segments"]], default=1),
    })
feats = pd.DataFrame(rows)

# Every sentence in every transcript, remembering which transcript it came from
sent_doc, sent_text = [], []
for i, d in enumerate(docs):
    for s in split_sentences(d["text"]):
        sent_doc.append(i)
        sent_text.append(s)
sent_doc = np.array(sent_doc)
print(len(sent_text), "sentences")

# ---------- 2. grammar errors via a grammar-correction model ----------
tok = AutoTokenizer.from_pretrained("grammarly/coedit-large")
gec = AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large", torch_dtype=torch.float16).to(device).eval()
corrected = []
for i in range(0, len(sent_text), 64):
    batch = ["Fix grammatical errors in this sentence: " + s for s in sent_text[i:i + 64]]
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = gec.generate(**enc, max_new_tokens=128, num_beams=1)
    corrected += tok.batch_decode(out, skip_special_tokens=True)
    if i % 1280 == 0:
        print("GEC", i, "|", sent_text[i], "->", corrected[-len(batch)])
del gec
torch.cuda.empty_cache()


def word_edits(a, b):
    """Number of words inserted/deleted/replaced to turn sentence a into b."""
    a, b = words_of(a), words_of(b)
    sm = difflib.SequenceMatcher(None, a, b)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


edits = np.array([word_edits(s, c) for s, c in zip(sent_text, corrected)])
sent_words = np.array([len(words_of(s)) for s in sent_text])

# ---------- 3. language-model surprise (GPT-2 loss per sentence) ----------
lm_tok = AutoTokenizer.from_pretrained("gpt2-medium")
lm = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device).eval()
sent_nll = []
for s in sent_text:
    ids = lm_tok(s, return_tensors="pt").input_ids[:, :256].to(device)
    if ids.shape[1] < 2:
        sent_nll.append(np.nan)
        continue
    with torch.no_grad():
        sent_nll.append(lm(ids, labels=ids).loss.item())
sent_nll = np.array(sent_nll)
del lm
torch.cuda.empty_cache()

# Aggregate sentence-level numbers back to one row per transcript
for i in range(len(docs)):
    m = sent_doc == i
    nw = max(sent_words[m].sum(), 1)
    feats.loc[i, "gec_edit_rate"] = edits[m].sum() / nw
    feats.loc[i, "gec_frac_sents_changed"] = (edits[m] > 0).mean() if m.any() else 0
    feats.loc[i, "gec_max_sent_edits"] = edits[m].max() if m.any() else 0
    feats.loc[i, "lm_mean_nll"] = np.nanmean(sent_nll[m]) if m.any() else np.nan
    feats.loc[i, "lm_max_nll"] = np.nanmax(sent_nll[m]) if m.any() else np.nan
    feats.loc[i, "lm_std_nll"] = np.nanstd(sent_nll[m]) if m.any() else np.nan

feats.to_csv(f"{OUT}/features.csv", index=False)
pd.DataFrame({"doc": [docs[i]["filename"] for i in sent_doc], "sentence": sent_text,
              "corrected": corrected, "edits": edits, "nll": sent_nll}).to_csv(f"{OUT}/sentences.csv", index=False)

# ---------- 4. transcript embeddings ----------
texts = [d["text"] if d["text"].strip() else "(silence)" for d in docs]
for name, model_id in [("mpnet", "sentence-transformers/all-mpnet-base-v2"),
                       ("bge", "BAAI/bge-large-en-v1.5")]:
    emb = SentenceTransformer(model_id, device=device).encode(texts, batch_size=32, normalize_embeddings=True,
                                                             show_progress_bar=False)
    np.save(f"{OUT}/emb_{name}.npy", emb)
    print(name, emb.shape)

print(feats.describe().T.to_string())
