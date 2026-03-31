import sys
import pandas as pd
import joblib
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="FASTA file containing peptide sequences")
args = parser.parse_args()

# Load model and feature list
rf = joblib.load("../models/bbb_rf_top200.pkl")
with open("../models/bbb_rf_top200_features.json") as f:
    features = json.load(f)

# Extract descriptors using iFeature
import subprocess
subprocess.run([
    "python", "../iFeature-master/iFeature.py",
    "--file", args.file,
    "--type", "AAC",  "--out", "tmp_AAC.tsv"
])

# Add ALL descriptor extractions here…

# Merge them (same pipeline as training)
# ...

# Load merged features
df = pd.read_csv("merged_tmp_features.csv")

# Subset to top-k features
df = df[features]

# Predict
prob = rf.predict_proba(df)[0][1]
pred = "BBB+" if prob > 0.5 else "BBB-"

print("\nPrediction:", pred)
print("Probability:", prob)
