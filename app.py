# ============================================================
# BBB-TransAI Streamlit App (working, using same pipeline as test_predict.py)
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from functools import reduce

# ------------------------------------------------------------
# PATHS & CONFIG
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
IFEATURE = ROOT / "iFeature-master" / "iFeature-master" / "iFeature.py"
MODEL_PATH = ROOT / "models" / "bbb_rf_top30.pkl"   # your trained RF

ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")

DESCRIPTORS = [
    "AAC", "DPC", "TPC", "APAAC",
    "CTDC", "CTDT", "CTDD",
    "CTriad", "NMBroto", "Moran", "Geary",
]


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(
            "Model does not have 'feature_names_in_'. "
            "You may need to re-train with a newer scikit-learn."
        )
    feature_names = list(model.feature_names_in_)
    return model, feature_names


# ------------------------------------------------------------
# HELPERS (same logic as working test_predict)
# ------------------------------------------------------------

def clean_seq(seq: str) -> str:
    """Uppercase, strip spaces, keep only valid AAs."""
    seq = seq.strip().upper().replace(" ", "")
    return "".join(a for a in seq if a in ALLOWED_AA)


def clean_ifeature_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize iFeature output:
    - ensure 'ID' column exists
    - drop duplicate column names
    - force ID to string
    """
    df.columns = [str(c).strip() for c in df.columns]

    if "ID" not in df.columns:
        if "#" in df.columns:
            df = df.rename(columns={"#": "ID"})
        else:
            first = df.columns[0]
            df = df.rename(columns={first: "ID"})

    df = df.loc[:, ~df.columns.duplicated()]
    df["ID"] = df["ID"].astype(str)
    return df


def run_ifeature(id_to_seq: dict) -> pd.DataFrame:
    """
    Write sequences to a temp FASTA, run iFeature for all 11 descriptors,
    and merge all TSVs on 'ID' using OUTER join (so we don't lose rows).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="bbbt_app_"))
    fasta_path = tmp_dir / "input.fasta"

    # write FASTA
    with open(fasta_path, "w") as f:
        for pid, seq in id_to_seq.items():
            f.write(f">{pid}\n{seq}\n")

    dfs = []

    for desc in DESCRIPTORS:
        out_tsv = tmp_dir / f"{fasta_path.stem}_{desc}.tsv"
        cmd = [
            sys.executable,
            str(IFEATURE),
            "--file", str(fasta_path),
            "--type", desc,
            "--out", str(out_tsv),
        ]
        # Show in terminal; Streamlit UI will have its own spinner
        print(f"Descriptor type: {desc}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"iFeature failed for {desc}:\n{res.stderr}")

        df = pd.read_csv(out_tsv, sep="\t")
        df = clean_ifeature_df(df)
        dfs.append(df)

    # OUTER merge all descriptor tables
    def merge_two(left, right):
        left = left.copy()
        right = right.copy()
        left["ID"] = left["ID"].astype(str)
        right["ID"] = right["ID"].astype(str)
        return pd.merge(left, right, on="ID", how="outer")

    merged = reduce(merge_two, dfs)

    # keep only the peptides we originally requested
    wanted_ids = {str(k) for k in id_to_seq.keys()}
    merged = merged[merged["ID"].isin(wanted_ids)].reset_index(drop=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Merged feature table shape:", merged.shape)
    return merged


def predict_from_features(feat: pd.DataFrame, model, feature_names):
    """
    Align iFeature table to model.feature_names_in_, fill missing with 0,
    and run RF prediction.
    """
    if feat.shape[0] == 0:
        raise ValueError("Merged feature table is empty; no valid samples to predict.")

    # make sure all expected features exist
    for col in feature_names:
        if col not in feat.columns:
            feat[col] = 0.0

    X = feat[feature_names].fillna(0.0).astype(float)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    out = pd.DataFrame({
        "ID": feat["ID"],
        "BBB_prob": probs,
        "BBB_pred": preds,
    })

    return out


def validate_id_to_seq(id_to_seq: dict):
    if not id_to_seq:
        return False, "No peptides provided."

    for pid, seq in id_to_seq.items():
        if len(seq) == 0:
            return False, f"Peptide {pid} has an empty sequence after cleaning."
        bad = set(seq) - ALLOWED_AA
        if bad:
            return False, f"Peptide {pid} has invalid residues: {bad}"

    return True, ""


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

def main():
    st.set_page_config(page_title="BBB-TransAI", layout="wide")
    st.title("🧠 BBB-TransAI – Blood–Brain Barrier Peptide Predictor")

    st.markdown(
        """
        Designed by Kartic, Lipid Biochemistry Lab.  
        """
    )

    model, feature_names = load_model()

    st.sidebar.header("Input Mode")
    mode = st.sidebar.radio("Choose Input Mode", ["Single peptide", "Batch pasted list"])

    st.sidebar.markdown("---")
    st.sidebar.write(f"Model file: `{MODEL_PATH.name}`")
    st.sidebar.write(f"Expected features: **{len(feature_names)}**")

    # ---------------- SINGLE PEPTIDE ----------------
    if mode == "Single peptide":
        st.subheader("Single peptide prediction")

        pid = st.text_input("Peptide ID:", "PEP_1")
        seq_raw = st.text_area("Peptide sequence:", height=120, value="YGRKKRRQRRR")

        if st.button("Predict BBB permeability"):
            seq_clean = clean_seq(seq_raw)
            id_to_seq = {pid: seq_clean}

            ok, msg = validate_id_to_seq(id_to_seq)
            if not ok:
                st.error(msg)
                return

            with st.spinner("Running iFeature and Random Forest model..."):
                feat = run_ifeature(id_to_seq)
                result = predict_from_features(feat, model, feature_names)

            st.success("Done!")
            st.dataframe(
                result.style.format({"BBB_prob": "{:.3f}"})
                .background_gradient(subset=["BBB_prob"], cmap="Greens")
            )

    # ---------------- BATCH MODE ----------------
    else:
        st.subheader("Batch peptide prediction")

        st.markdown(
            """
            Paste multiple peptides, **one per line**, in the format:
            ```text
            P1 YGRKKRRQRRR
            P2 TFFYGGSRGKRNNFKTEEY
            P3 LRKLRKRLLR
            ```
            """
        )

        text_block = st.text_area("Peptide list:", height=240)

        if st.button("Predict BBB for all peptides"):
            id_to_seq = {}
            for line in text_block.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                pid = parts[0]
                seq = "".join(parts[1:])
                id_to_seq[pid] = clean_seq(seq)

            ok, msg = validate_id_to_seq(id_to_seq)
            if not ok:
                st.error(msg)
                return

            with st.spinner("Running iFeature and Random Forest model for all peptides..."):
                feat = run_ifeature(id_to_seq)
                result = predict_from_features(feat, model, feature_names)

            st.success("Done!")
            result_sorted = result.sort_values("BBB_prob", ascending=False)

            st.dataframe(
                result_sorted.style.format({"BBB_prob": "{:.3f}"})
                .background_gradient(subset=["BBB_prob"], cmap="Greens")
            )

            st.subheader("BBB probability bar plot")
            st.bar_chart(result_sorted.set_index("ID")["BBB_prob"])


if __name__ == "__main__":
    main()
