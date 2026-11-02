# Step 3: an LLM acts as the examiner.
# Each transcript is shown to an open LLM together with the official grammar rubric.
# Instead of letting it write free text, we read its probabilities for the answers
# "1".."5" and take the expected value, which gives a smooth score (e.g. 3.4).

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], check=True)

import glob, json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

docs = json.load(open(glob.glob("/kaggle/input/**/transcripts.json", recursive=True)[0]))

RUBRIC = """Grammar score rubric:
1 - Struggles with proper sentence structure and syntax; limited control over simple grammatical structures and memorized sentence patterns.
2 - Limited understanding of sentence structure and syntax. Uses simple structures but consistently makes basic sentence structure and grammatical mistakes; may leave sentences incomplete.
3 - Decent grasp of sentence structure but makes errors in grammatical structure, or decent grasp of grammatical structure but errors in sentence syntax and structure.
4 - Strong understanding of sentence structure and syntax; consistently good control of grammar. Occasional minor errors that do not cause misunderstanding; can correct most of them.
5 - High grammatical accuracy and adept control of complex grammar. Uses grammar accurately and effectively, seldom making noticeable mistakes; handles complex structures well and self-corrects when necessary."""

PROMPT = """You are an experienced English speaking examiner. Below is an automatic transcript of a candidate speaking for about one minute. Fillers (um, uh) and punctuation come from the speech-to-text system, so judge the candidate's grammar, not the punctuation.

{rubric}

Transcript:
\"\"\"{text}\"\"\"

What grammar score (1-5) does this candidate deserve? Answer with a single digit."""

MODELS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"]
out = pd.DataFrame({"filename": [d["filename"] for d in docs], "split": [d["split"] for d in docs]})

for model_id in MODELS:
    short = model_id.split("/")[-1].lower().replace(".", "")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda:0",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_quant_type="nf4"))
    except Exception as e:  # e.g. model not supported by the installed transformers version
        print("skipping", model_id, "->", repr(e)[:300])
        continue
    model.eval()
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 6)]

    probs = []
    for i, d in enumerate(docs):
        msgs = [{"role": "user", "content": PROMPT.format(rubric=RUBRIC, text=d["text"].strip() or "(no speech)")}]
        kw = {"enable_thinking": False} if "qwen3" in short else {}
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw) + "Grammar score: "
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
        with torch.no_grad():
            logits = model(ids).logits[0, -1, digit_ids].float()
        p = torch.softmax(logits, -1).cpu().numpy()
        probs.append(p)
        if i % 100 == 0:
            print(short, i, np.round(p, 2), "expected", round(float(p @ np.arange(1, 6)), 2))
    probs = np.array(probs)
    out[f"llm_{short}_score"] = probs @ np.arange(1, 6)
    for k in range(5):
        out[f"llm_{short}_p{k + 1}"] = probs[:, k]
    out.to_csv("/kaggle/working/llm_scores.csv", index=False)
    del model
    torch.cuda.empty_cache()

print(out.describe().T.to_string())
