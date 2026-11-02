# Grammar Scoring Engine for Spoken English

Solution to the **SHL Hiring Assessment 2026** Kaggle challenge: predict the grammar score (1–5) that human raters gave a 45–60 s recording of someone speaking English.

**Result:** public leaderboard RMSE **0.3465**. Speaker-grouped cross-validation: RMSE 0.547, Pearson 0.841.

The full report (approach, preprocessing, pipeline, evaluation, charts, training RMSE) is in **[`notebook/grammar_scoring_engine.ipynb`](notebook/grammar_scoring_engine.ipynb)**.

## Approach

1. **Speech → text.** Whisper large-v3, prompted towards verbatim output so fillers and grammar mistakes are kept.
2. **Text features.** Grammar-error rate from a grammar corrector (CoEdIT-large), GPT-2 sentence surprise, ASR confidence, fluency from word timings, an **LLM examiner** (Qwen) that scores the transcript against the official rubric, and text embeddings.
3. **Audio features.** WavLM-large and Whisper-encoder embeddings (first 45 s, to match the shorter test clips), plus a Whisper encoder **fine-tuned** on the scores.
4. **Model.** Ridge / LightGBM base models combined by a non-negative linear stack.
5. **Validation.** Many training speakers appear more than once; test speakers don't. All models are therefore validated with **speaker-grouped folds** (clips clustered by voice similarity), which made CV agree with the leaderboard.

## Repository layout

| Path | Contents |
|---|---|
| `notebook/grammar_scoring_engine.ipynb` | Report notebook (executed, with outputs) |
| `src/pipeline.py` | Final modelling pipeline: loads features, trains base models with grouped CV, stacks |
| `src/predict.py` | Writes `submission.csv` with the final pipeline |
| `kernels/` | GPU steps, run as Kaggle scripts (each folder has its script + `kernel-metadata.json`) |
| `experiments/` | Earlier exploration scripts (random-fold CV; kept for reference) |

GPU steps, in order:

| Script | Output (downloaded to `outputs/<name>/`) |
|---|---|
| `kernels/transcribe/transcribe.py` | `asr/transcripts.json`: transcripts, word timings |
| `kernels/features/features.py` | `features/`: hand features, sentence-level corrections, text embeddings |
| `kernels/llm_judge/llm_judge.py` | `llm-judge/llm_scores.csv`: LLM-examiner scores |
| `kernels/audio2/audio_emb2.py` | `audio2/`: 60 s audio embeddings (layer 6 is used to build the voice groups) |
| `kernels/audio45/audio_emb45.py` | `audio45/`: 45 s audio embeddings used by the models |
| `kernels/finetune_whisper_g/finetune_whisper_g.py` | `finetune-whisper-g/`: fine-tuned Whisper predictions (grouped folds) |

`kernels/audio`, `kernels/finetune` and `kernels/finetune_whisper` are earlier variants used in intermediate submissions.

## Reproducing

The competition data and everything derived from it are **not** in this repository (the competition rules forbid redistributing the data).

1. Join the competition and download the data into `Dataset_Final/`.
2. Run the Kaggle scripts above in order (`kaggle kernels push -p kernels/<name>`, with your username in each `kernel-metadata.json`) and download each output into `outputs/<name>/` with `kaggle kernels output`.
3. Local modelling (Python 3.12):
   ```bash
   python -m venv .venv && .venv/bin/pip install -r requirements.txt
   .venv/bin/python src/predict.py                       # writes submission.csv
   .venv/bin/jupyter nbconvert --execute --to notebook --inplace notebook/grammar_scoring_engine.ipynb
   ```
