"""Train the final pipeline and write the submission file.

Usage (from the repository root): python src/predict.py [--out submission.csv]
"""
import argparse

from pipeline import run

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="submission.csv")
args = ap.parse_args()

result = run()
result["submission"].to_csv(args.out, index=False)
print("wrote", args.out)
