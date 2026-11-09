"""
BBB-TransAI prediction core.

Sequence cleaning, iFeature descriptor extraction, structural descriptors,
the random-forest prediction and an exact additive decomposition of each
prediction into per-descriptor contributions.

Nothing in here imports a web framework, so it can be used from the server,
from a notebook or from a script.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

IFEATURE = ROOT / "iFeature-master" / "iFeature-master" / "iFeature.py"
MODEL_PATH = ROOT / "models" / "bbb_rf_top30.pkl"
DATASET_PATH = ROOT / "data" / "bbb_dataset.csv"
STRUCT_FEATURES_PATH = ROOT / "features" / "bbb_struct_features_100.csv"

ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")

DESCRIPTORS = [
    "AAC", "DPC", "TPC", "APAAC",
    "CTDC", "CTDT", "CTDD",
    "CTriad", "NMBroto", "Moran", "Geary",
]

# The three descriptors in the published 30-feature model that are derived from
# the 3D structure rather than from iFeature. Serving the model without them is
# not the same model, so they are always supplied: measured when the user
# uploads a structure, estimated from the sequence otherwise.
STRUCTURAL_FEATURES = ("radius_gyration", "compactness", "frac_positive")

POSITIVE_AA = set("KRH")
NEGATIVE_AA = set("DE")
HYDROPHOBIC_AA = set("AVILMFWYP")
AROMATIC_AA = set("FYW")

# Fallback radius-of-gyration scaling, used only if the reference structural
# feature table is missing. Refitted from that table at import when present.
RG_SCALE_DEFAULT = 2.2339
RG_EXPONENT_DEFAULT = 0.5029

MAX_SEQ_LENGTH = 200
MAX_PEPTIDES = 200


class PipelineError(RuntimeError):
    """Raised for conditions a user can act on, and shown to them verbatim."""


# ---------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------

def clean_seq(seq: str) -> str:
    """Uppercase, drop whitespace and keep only the 20 standard residues."""
    seq = "".join(seq.split()).upper()
    return "".join(a for a in seq if a in ALLOWED_AA)


def parse_batch(text: str) -> list[tuple[str, str]]:
    """
    Parse a pasted block into (id, sequence) pairs.

    Accepts FASTA, "ID SEQUENCE" / "ID,SEQUENCE" per line, or a bare sequence
    per line, which is given a generated identifier.
    """
    pairs: list[tuple[str, str]] = []
    pending_id: str | None = None
    pending_seq: list[str] = []

    def flush() -> None:
        nonlocal pending_id, pending_seq
        if pending_id is not None:
            pairs.append((pending_id, "".join(pending_seq)))
        pending_id, pending_seq = None, []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith(">"):
            flush()
            pending_id = line[1:].strip().split()[0] if line[1:].strip() else ""
            continue

        if pending_id is not None:
            # inside a FASTA record: lines of pure sequence continue it
            if set(line.upper()) <= ALLOWED_AA | {"-", "*"}:
                pending_seq.append(line)
                continue
            flush()

        parts = [p for p in line.replace(",", " ").replace("\t", " ").split() if p]
        if len(parts) >= 2:
            pairs.append((parts[0], "".join(parts[1:])))
        else:
            pairs.append(("", parts[0]))

    flush()

    out: list[tuple[str, str]] = []
    for index, (pid, seq) in enumerate(pairs, start=1):
        pid = pid.strip() or f"PEP_{index}"
        out.append((pid, seq))
    return out


def validate(id_to_seq: dict[str, str]) -> None:
    if not id_to_seq:
        raise PipelineError("No peptide sequences found in the input.")
    if len(id_to_seq) > MAX_PEPTIDES:
        raise PipelineError(
            f"{len(id_to_seq)} peptides submitted; this server accepts at most {MAX_PEPTIDES} at a time."
        )
    for pid, seq in id_to_seq.items():
        if not seq:
            raise PipelineError(f"{pid}: no valid residues left after cleaning the sequence.")
        if len(seq) < 2:
            raise PipelineError(f"{pid}: a sequence needs at least 2 residues.")
        if len(seq) > MAX_SEQ_LENGTH:
            raise PipelineError(
                f"{pid}: {len(seq)} residues, longer than the {MAX_SEQ_LENGTH}-residue limit."
            )


def dedupe_ids(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Build an id -> sequence map, suffixing ids that repeat."""
    out: dict[str, str] = {}
    for pid, seq in pairs:
        candidate, n = pid, 2
        while candidate in out:
            candidate = f"{pid}_{n}"
            n += 1
        out[candidate] = seq
    return out


# ---------------------------------------------------------------------------
# composition and structural descriptors
# ---------------------------------------------------------------------------

def _fit_rg_scaling() -> tuple[float, float, dict]:
    """
    Fit radius of gyration against chain length on the reference structures
    that ship with the repository, so the estimator used at inference is
    derived from data in the repository rather than from constants pasted in.
    """
    meta = {"source": "default constants", "n": 0}
    try:
        table = pd.read_csv(STRUCT_FEATURES_PATH)
        length = table["length"].to_numpy(float)
        rg = table["radius_gyration"].to_numpy(float)
        keep = (length > 0) & (rg > 0) & np.isfinite(length) & np.isfinite(rg)
        length, rg = length[keep], rg[keep]
        if len(length) < 10:
            raise ValueError("too few reference structures to fit")
        exponent, log_scale = np.polyfit(np.log(length), np.log(rg), 1)
        scale = float(np.exp(log_scale))
        predicted = scale * length ** exponent
        meta = {
            "source": STRUCT_FEATURES_PATH.name,
            "n": int(len(length)),
            "mean_abs_error_angstrom": round(float(np.abs(predicted - rg).mean()), 3),
            "median_rel_error": round(float(np.median(np.abs(predicted - rg) / rg)), 4),
        }
        return scale, float(exponent), meta
    except Exception:
        return RG_SCALE_DEFAULT, RG_EXPONENT_DEFAULT, meta


RG_SCALE, RG_EXPONENT, RG_FIT_META = _fit_rg_scaling()


def estimate_radius_of_gyration(length: int) -> float:
    """Rg estimated from chain length by the fitted power law."""
    return float(RG_SCALE * (length ** RG_EXPONENT))


def composition(seq: str) -> dict:
    """Sequence-derived composition descriptors. All exact."""
    length = len(seq)
    positive = sum(1 for a in seq if a in POSITIVE_AA)
    negative = sum(1 for a in seq if a in NEGATIVE_AA)
    return {
        "length": length,
        "frac_positive": positive / length,
        "frac_negative": negative / length,
        "net_charge": positive - negative,
        "frac_hydrophobic": sum(1 for a in seq if a in HYDROPHOBIC_AA) / length,
        "frac_aromatic": sum(1 for a in seq if a in AROMATIC_AA) / length,
    }


def structural_descriptors(seq: str, radius_gyration: float | None = None) -> tuple[dict, str]:
    """
    The three structure-derived model inputs.

    frac_positive is exact from the sequence. Radius of gyration is measured
    when a structure was supplied and estimated from length otherwise;
    compactness is length / Rg either way.
    """
    comp = composition(seq)
    if radius_gyration is not None and radius_gyration > 0:
        rg, mode = float(radius_gyration), "measured"
    else:
        rg, mode = estimate_radius_of_gyration(comp["length"]), "estimated"
    return {
        "radius_gyration": rg,
        "compactness": comp["length"] / rg,
        "frac_positive": comp["frac_positive"],
    }, mode


def radius_of_gyration_from_structure(text: str) -> tuple[float, int]:
    """
    Heavy-atom radius of gyration from a PDB or mmCIF file.

    Matches the reference implementation in notebooks/structural_features.ipynb:
    all non-hydrogen atoms of the first model, equal weights.
    Returns (Rg in angstrom, residue count).
    """
    coords: list[tuple[float, float, float]] = []
    residues: set[tuple[str, str]] = set()

    lines = text.splitlines()
    is_cif = any(line.startswith(("data_", "loop_", "_atom_site.")) for line in lines[:400])

    if is_cif:
        columns: list[str] = []
        in_atom_loop = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("_atom_site."):
                columns.append(stripped.split(".", 1)[1].split()[0])
                in_atom_loop = True
                continue
            if in_atom_loop:
                if stripped.startswith(("#", "loop_", "_")) or not stripped:
                    if columns:
                        break
                    continue
                fields = stripped.split()
                if len(fields) < len(columns):
                    continue
                row = dict(zip(columns, fields))
                if row.get("group_PDB", "ATOM") not in ("ATOM", "HETATM"):
                    continue
                element = (row.get("type_symbol") or "").upper()
                if element == "H":
                    continue
                model = row.get("pdbx_PDB_model_num")
                if model not in (None, ".", "1"):
                    continue
                try:
                    coords.append((
                        float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"]),
                    ))
                except (KeyError, ValueError):
                    continue
                residues.add((row.get("label_asym_id", "A"), row.get("label_seq_id", "")))
    else:
        for line in lines:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            element = line[76:78].strip().upper()
            name = line[12:16].strip()
            if element == "H" or (not element and name.startswith("H")):
                continue
            try:
                coords.append((
                    float(line[30:38]), float(line[38:46]), float(line[46:54]),
                ))
            except ValueError:
                continue
            residues.add((line[21:22], line[22:27].strip()))

    if len(coords) < 3:
        raise PipelineError(
            "No heavy atoms could be read from that file. A PDB or mmCIF model is expected."
        )

    xyz = np.asarray(coords, dtype=float)
    centre = xyz.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((xyz - centre) ** 2, axis=1))))
    return rg, len(residues)


# ---------------------------------------------------------------------------
# iFeature
# ---------------------------------------------------------------------------

def ifeature_available() -> bool:
    return IFEATURE.is_file()


def _clean_ifeature_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(c).strip() for c in frame.columns]
    if "ID" not in frame.columns:
        first = "#" if "#" in frame.columns else frame.columns[0]
        frame = frame.rename(columns={first: "ID"})
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame["ID"] = frame["ID"].astype(str)
    return frame


def run_ifeature(
    id_to_seq: dict[str, str],
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """
    Write the peptides to a temporary FASTA, run every descriptor family and
    outer-join the tables on ID. `progress` is called as each family finishes.
    """
    if not ifeature_available():
        raise PipelineError(
            "iFeature is not installed next to the application. Download iFeature and place "
            f"it so that {IFEATURE.relative_to(ROOT)} exists, then restart."
        )

    tmp = Path(tempfile.mkdtemp(prefix="bbbtransai_"))
    try:
        fasta = tmp / "input.fasta"
        with open(fasta, "w", encoding="utf8") as handle:
            for pid, seq in id_to_seq.items():
                handle.write(f">{pid}\n{seq}\n")

        tables = []
        total = len(DESCRIPTORS)
        for index, descriptor in enumerate(DESCRIPTORS, start=1):
            if progress:
                progress(index - 1, total, descriptor)
            out_tsv = tmp / f"{fasta.stem}_{descriptor}.tsv"
            result = subprocess.run(
                [sys.executable, str(IFEATURE), "--file", str(fasta),
                 "--type", descriptor, "--out", str(out_tsv)],
                capture_output=True, text=True,
            )
            if result.returncode != 0 or not out_tsv.exists():
                detail = (result.stderr or result.stdout or "").strip().splitlines()
                raise PipelineError(
                    f"iFeature failed on descriptor {descriptor}: "
                    + (detail[-1] if detail else "no output produced")
                )
            tables.append(_clean_ifeature_table(pd.read_csv(out_tsv, sep="\t")))
            if progress:
                progress(index, total, descriptor)

        def join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
            return pd.merge(left, right, on="ID", how="outer")

        merged = reduce(join, tables)
        wanted = {str(k) for k in id_to_seq}
        merged = merged[merged["ID"].isin(wanted)].reset_index(drop=True)
        if merged.empty:
            raise PipelineError("iFeature returned no rows for the submitted peptides.")
        return merged
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@dataclass
class Model:
    estimator: object
    feature_names: list[str]
    importances: dict[str, float]
    sequence_features: list[str] = field(default_factory=list)


_model: Model | None = None


def load_model() -> Model:
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.is_file():
        raise PipelineError(f"Model file not found: {MODEL_PATH}")
    estimator = joblib.load(MODEL_PATH)
    if not hasattr(estimator, "feature_names_in_"):
        raise PipelineError(
            "The model was saved without feature names; retrain with scikit-learn 1.0 or newer."
        )
    names = [str(n) for n in estimator.feature_names_in_]
    _model = Model(
        estimator=estimator,
        feature_names=names,
        importances={n: float(v) for n, v in zip(names, estimator.feature_importances_)},
        sequence_features=[n for n in names if n not in STRUCTURAL_FEATURES],
    )
    return _model


def contributions(forest, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose each predicted probability into a base rate plus one additive
    contribution per descriptor.

    For every tree the decision path is walked and the change in P(class=1)
    at each split is credited to the descriptor that was split on. Averaged
    over the forest this is exact: base + contributions.sum() == predict_proba.
    """
    values = np.asarray(matrix, dtype=np.float32)
    n_samples, n_features = values.shape
    credit = np.zeros((n_samples, n_features), dtype=float)
    base = np.zeros(n_samples, dtype=float)

    for estimator in forest.estimators_:
        tree = estimator.tree_
        node_value = tree.value[:, 0, :]
        totals = node_value.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        positive = (node_value / totals)[:, 1]
        feature_of = tree.feature

        paths = estimator.decision_path(values)
        indices, pointers = paths.indices, paths.indptr
        for row in range(n_samples):
            nodes = indices[pointers[row]:pointers[row + 1]]
            base[row] += positive[nodes[0]]
            for parent, child in zip(nodes[:-1], nodes[1:]):
                credit[row, feature_of[parent]] += positive[child] - positive[parent]

    n_trees = len(forest.estimators_)
    return base / n_trees, credit / n_trees


# ---------------------------------------------------------------------------
# reference distribution
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    n_peptides: int
    n_positive: int
    metrics: dict[str, list[float]]

    def percentile(self, metric: str, value: float) -> float | None:
        series = self.metrics.get(metric)
        if not series:
            return None
        return round(100.0 * float(np.searchsorted(series, value, side="right")) / len(series), 1)


_reference: Reference | None = None


def load_reference() -> Reference | None:
    """Composition distribution of the training peptides, for context readouts."""
    global _reference
    if _reference is not None:
        return _reference
    try:
        table = pd.read_csv(DATASET_PATH)
        sequences = [clean_seq(str(s)) for s in table["Sequence"]]
        sequences = [s for s in sequences if s]
        rows = [composition(s) for s in sequences]
        metrics = {
            key: sorted(float(r[key]) for r in rows)
            for key in ("length", "frac_positive", "net_charge", "frac_hydrophobic", "frac_aromatic")
        }
        metrics["radius_gyration"] = sorted(estimate_radius_of_gyration(r["length"]) for r in rows)
        labels = table["BBB_label"] if "BBB_label" in table.columns else None
        _reference = Reference(
            n_peptides=len(sequences),
            n_positive=int(labels.sum()) if labels is not None else 0,
            metrics=metrics,
        )
    except Exception:
        _reference = None
    return _reference


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

def predict(
    pairs: Iterable[tuple[str, str]],
    structures: dict[str, float] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Run the full pipeline for a set of peptides.

    `structures` maps peptide id to a measured radius of gyration; ids absent
    from it fall back to the length-based estimate.
    """
    structures = structures or {}
    id_to_seq = dedupe_ids((pid, clean_seq(seq)) for pid, seq in pairs)
    validate(id_to_seq)

    model = load_model()
    reference = load_reference()

    table = run_ifeature(id_to_seq, progress=progress)
    table = table.set_index("ID")

    missing_from_ifeature = [c for c in model.sequence_features if c not in table.columns]
    if missing_from_ifeature:
        raise PipelineError(
            "iFeature did not produce "
            f"{len(missing_from_ifeature)} descriptor(s) the model requires, including "
            f"{missing_from_ifeature[0]}. The iFeature installation may be incomplete."
        )

    ordered_ids = list(id_to_seq)
    rows, modes = [], {}
    for pid in ordered_ids:
        if pid not in table.index:
            raise PipelineError(f"iFeature returned no descriptors for {pid}.")
        row = table.loc[pid, model.sequence_features].astype(float).to_dict()
        structural, mode = structural_descriptors(id_to_seq[pid], structures.get(pid))
        row.update(structural)
        rows.append(row)
        modes[pid] = mode

    frame = pd.DataFrame(rows, index=ordered_ids).reindex(columns=model.feature_names)
    if frame.isna().any().any():
        bad = frame.columns[frame.isna().any()].tolist()
        raise PipelineError(f"Missing descriptor values for: {', '.join(bad[:5])}")

    matrix = frame.to_numpy(dtype=float)
    probabilities = model.estimator.predict_proba(frame)[:, 1]
    base, credit = contributions(model.estimator, matrix)

    results = []
    for index, pid in enumerate(ordered_ids):
        seq = id_to_seq[pid]
        comp = composition(seq)
        structural, _ = structural_descriptors(seq, structures.get(pid))
        comp["radius_gyration"] = structural["radius_gyration"]
        comp["compactness"] = structural["compactness"]

        order = np.argsort(-np.abs(credit[index]))[:8]
        drivers = [
            {
                "feature": model.feature_names[j],
                "label": describe_feature(model.feature_names[j]),
                "contribution": round(float(credit[index, j]), 5),
                "value": round(float(matrix[index, j]), 4),
                "importance": round(model.importances[model.feature_names[j]], 5),
            }
            for j in order
        ]

        probability = float(probabilities[index])
        results.append({
            "id": pid,
            "sequence": seq,
            "probability": round(probability, 4),
            "permeable": bool(probability >= threshold),
            "structure_mode": modes[pid],
            "composition": {
                "length": comp["length"],
                "net_charge": comp["net_charge"],
                "frac_positive": round(comp["frac_positive"], 4),
                "frac_negative": round(comp["frac_negative"], 4),
                "frac_hydrophobic": round(comp["frac_hydrophobic"], 4),
                "frac_aromatic": round(comp["frac_aromatic"], 4),
                "radius_gyration": round(comp["radius_gyration"], 3),
                "compactness": round(comp["compactness"], 3),
            },
            "percentiles": {
                key: reference.percentile(key, comp[key]) if reference else None
                for key in ("length", "net_charge", "frac_positive", "frac_hydrophobic")
            },
            "base_rate": round(float(base[index]), 4),
            "drivers": drivers,
        })

    return {
        "threshold": threshold,
        "results": results,
        "model": model_card(),
    }


# ---------------------------------------------------------------------------
# descriptor labels and model card
# ---------------------------------------------------------------------------

_PROPERTY_LABELS = {
    "hydrophobicity": "hydrophobicity",
    "secondarystruct": "secondary structure",
    "polarity": "polarity",
    "polarizability": "polarisability",
    "solventaccess": "solvent accessibility",
    "charge": "charge",
    "normwaalsvolume": "van der Waals volume",
}

_AAINDEX_LABELS = {
    "CHOC760101": "residue accessible surface area",
    "CHAM820101": "polarisability",
    "CHAM820102": "free energy of solution",
    "CIDH920105": "normalised hydrophobicity",
}

_PLAIN_LABELS = {
    "radius_gyration": "Radius of gyration",
    "compactness": "Compactness (length / Rg)",
    "frac_positive": "Fraction of K, R, H",
}


def describe_feature(name: str) -> str:
    """Turn an iFeature column name into something readable in the interface."""
    if name in _PLAIN_LABELS:
        return _PLAIN_LABELS[name]

    head, _, tail = name.partition(".")
    base = head.split("_")[0]
    index_code = head.split("_")[1] if "_" in head else None

    if base in _AAINDEX_LABELS or head in _AAINDEX_LABELS:
        code = head.split("_")[0]
        label = _AAINDEX_LABELS.get(code, code)
        if tail.startswith("lag"):
            lag = tail.replace("lag", "").split("_")[0]
            return f"Autocorrelation of {label}, lag {lag}"
        return f"Autocorrelation of {label}"

    prop = _PROPERTY_LABELS.get(base)
    if prop is None:
        return name

    detail = f" ({_AAINDEX_LABELS.get(index_code, index_code)})" if index_code and index_code in _AAINDEX_LABELS else ""
    scale = f" [{index_code}]" if index_code and not detail else ""

    if tail.startswith("G"):
        return f"{prop.capitalize()}{scale}{detail}: fraction in group {tail[1:]}"
    if tail.startswith("Tr"):
        return f"{prop.capitalize()}{scale}{detail}: transitions between groups {tail[2:3]} and {tail[3:4]}"
    if ".residue" in f".{tail}":
        group, _, rest = tail.partition(".")
        pct = rest.replace("residue", "")
        return f"{prop.capitalize()}{scale}{detail}: position where {pct}% of group {group} is reached"
    return f"{prop.capitalize()}{scale}{detail}: {tail}"


def model_card() -> dict:
    model = load_model()
    reference = load_reference()
    estimator = model.estimator
    top = sorted(model.importances.items(), key=lambda kv: -kv[1])
    return {
        "name": MODEL_PATH.name,
        "kind": type(estimator).__name__,
        "n_trees": int(getattr(estimator, "n_estimators", 0)),
        "n_features": len(model.feature_names),
        "n_sequence_features": len(model.sequence_features),
        "structural_features": list(STRUCTURAL_FEATURES),
        "descriptor_families": DESCRIPTORS,
        "training_peptides": reference.n_peptides if reference else None,
        "training_positive": reference.n_positive if reference else None,
        "rg_estimator": {
            "scale": round(RG_SCALE, 4),
            "exponent": round(RG_EXPONENT, 4),
            **RG_FIT_META,
        },
        "top_features": [
            {"feature": f, "label": describe_feature(f), "importance": round(v, 5)}
            for f, v in top[:10]
        ],
        "structural_importance": round(
            sum(model.importances[f] for f in STRUCTURAL_FEATURES if f in model.importances), 4
        ),
    }


def status() -> dict:
    """Everything the interface needs to describe the server at load."""
    info = {
        "ifeature": ifeature_available(),
        "ifeature_path": str(IFEATURE),
        "model_present": MODEL_PATH.is_file(),
        "max_peptides": MAX_PEPTIDES,
        "max_length": MAX_SEQ_LENGTH,
        "allowed_residues": "".join(sorted(ALLOWED_AA)),
    }
    try:
        info["model"] = model_card()
    except PipelineError as exc:
        info["model"] = None
        info["error"] = str(exc)
    return info


if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
