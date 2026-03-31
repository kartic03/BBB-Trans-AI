import argparse
import os
import pandas as pd
from functools import reduce

# All 11 descriptor families
DESCRIPTORS = [
    "AAC",
    "DPC",
    "TPC",
    "APAAC",
    "CTDC",
    "CTDT",
    "CTDD",
    "CTriad",
    "NMBroto",
    "Moran",
    "Geary",
]

def load_descriptor(path: str) -> pd.DataFrame:
    """Load a descriptor TSV and clean ID + duplicate columns."""
    if not os.path.exists(path):
        print(f"[WARN] File not found, skipping: {path}")
        return None

    df = pd.read_csv(path, sep="\t")

    if df.shape[1] < 2:
        print(f"[WARN] Descriptor file has <2 columns, skipping: {path}")
        return None

    # First column is the sequence ID in iFeature outputs
    first_col = df.columns[0]

    # Rename the first column to ID
    df = df.rename(columns={first_col: "ID"})

    # Convert ID to string
    df["ID"] = df["ID"].astype(str)

    # Remove any duplicated column names (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Extra safety: if multiple 'ID' columns somehow remain, keep only the first
    id_indices = [i for i, c in enumerate(df.columns) if c == "ID"]
    if len(id_indices) > 1:
        keep_idx = id_indices[0]
        cols = list(df.columns)
        for i in id_indices[1:]:
            cols[i] = f"ID_dup_{i}"
        df.columns = cols
        # Drop any ID_dup columns just in case
        df = df.drop(columns=[c for c in df.columns if c.startswith("ID_dup_")])

    print(f"[INFO] Loaded {os.path.basename(path)} with shape {df.shape}")
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple iFeature descriptor files into one feature table."
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix of descriptor files (e.g. B3 or ME). "
             "Script expects files like B3_AAC.tsv, B3_DPC.tsv, ... in ../features/",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path, e.g. ../features/bbb_all_features_11desc.csv",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(base_dir, "..", "features")

    dfs = []
    for desc in DESCRIPTORS:
        fname = f"{args.prefix}_{desc}.tsv"
        path = os.path.join(features_dir, fname)
        df = load_descriptor(path)
        if df is not None:
            dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError("No descriptor files were loaded; nothing to merge.")

    def merge_two(left, right):
        return pd.merge(left, right, on="ID", how="inner")

    merged = reduce(merge_two, dfs)
    print(f"[INFO] Final merged feature shape: {merged.shape}")

    # Resolve output path (relative vs absolute)
    if os.path.isabs(args.out):
        out_path = args.out
    else:
        out_path = os.path.join(base_dir, "..", args.out)

    merged.to_csv(out_path, index=False)
    print(f"[INFO] Saved merged features to: {out_path}")

if __name__ == "__main__":
    main()
