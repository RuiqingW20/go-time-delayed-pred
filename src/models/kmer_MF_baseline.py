#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
k-mer (char-level) + One-vs-Rest LogisticRegression baseline for MF multilabel GO prediction.

Input:
  data/processed/MF_dataset.parquet with columns:
    - accession (str)
    - sequence (str)
    - labels (list[str])
    - n_labels (int)

Output:
  - joblib model bundle (vectorizer + classifier + binarizer + threshold info)
  - JSON metrics summary

Example:
  python src/baselines/kmers_lr_mf.py \
    --data data/processed/MF_dataset.parquet \
    --outdir runs/kmers_lr_mf \
    --k 3 --min_df 2 --max_features 200000 \
    --C 2.0 --n_jobs -1 --seed 42 \
    --threshold_strategy global
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, average_precision_score

from pathlib import Path
import pandas as pd
import sys
import pyarrow.parquet as pq
# p = Path("data/processed/MF_dataset.parquet").resolve()
# print("ABS PATH:", p)
# print("EXISTS:", p.exists())
# print("SIZE BYTES:", p.stat().st_size if p.exists() else None)
# pf = pq.ParquetFile(p)
# print("num_row_groups:", pf.num_row_groups)
# print("schema:", pf.schema)
# df = pd.read_parquet(p)
# print("READ rows:", len(df))
# print("COLUMNS:", df.columns.tolist())
# print("DTYPES:\n", df.dtypes)
# print("HEAD:\n", df.head(3))
# sys.exit(0)
# ----------------------------
# Repro helpers
# ----------------------------
def set_global_seed(seed: int) -> np.random.Generator:
    # sklearn itself uses random_state passed into estimators;
    # here we control numpy RNG for splitting, etc.
    return np.random.default_rng(seed)


# ----------------------------
# Split strategy
# ----------------------------
def train_val_split(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple protein-level random split.

    Reasonable baseline split for "is dataset learnable".
    Later (for stronger baselines) you can do:
      - taxon-aware split
      - sequence-identity cluster split (avoid homolog leakage)
    """
    rng = set_global_seed(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n_val = int(round(len(df) * val_ratio))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[val_idx].reset_index(drop=True)
    return df_tr, df_va


def summarize_split(df_tr: pd.DataFrame, df_va: pd.DataFrame, y_tr: np.ndarray, y_va: np.ndarray) -> Dict:
    def stats(df: pd.DataFrame, y: np.ndarray) -> Dict:
        return {
            "n_samples": int(len(df)),
            "n_labels_total": int(y.shape[1]),
            "label_cardinality_mean": float(df["n_labels"].mean()),
            "label_cardinality_median": float(df["n_labels"].median()),
            "positives_per_label_mean": float(y.sum(axis=0).mean()),
            "positives_per_label_median": float(np.median(y.sum(axis=0))),
        }

    return {"train": stats(df_tr, y_tr), "val": stats(df_va, y_va)}


# ----------------------------
# Threshold selection
# ----------------------------
def select_global_threshold_by_micro_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid: np.ndarray,
) -> Tuple[float, Dict]:
    best_t = 0.5
    best = -1.0
    curve = []

    for t in grid:
        y_pred = (y_score >= t).astype(np.int8)
        score = f1_score(y_true, y_pred, average="micro", zero_division=0)
        curve.append((float(t), float(score)))
        if score > best:
            best = score
            best_t = float(t)

    info = {"best_micro_f1": float(best), "grid_size": int(len(grid)), "curve_sampled": curve[:: max(1, len(curve)//50)]}
    return best_t, info


def select_per_label_thresholds_by_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid: np.ndarray,
    min_pos: int = 5,
) -> Tuple[np.ndarray, Dict]:
    """
    Per-label threshold tuning on val: choose t_j maximizing F1 for label j.
    Guardrail: if a label has too few positives in val, fall back to 0.5.
    """
    n_labels = y_true.shape[1]
    thr = np.full(n_labels, 0.5, dtype=np.float32)
    chosen = 0
    skipped = 0

    for j in range(n_labels):
        if int(y_true[:, j].sum()) < min_pos:
            skipped += 1
            continue

        best_t = 0.5
        best = -1.0
        yt = y_true[:, j]
        ys = y_score[:, j]
        for t in grid:
            yp = (ys >= t).astype(np.int8)
            s = f1_score(yt, yp, average="binary", zero_division=0)
            if s > best:
                best = s
                best_t = float(t)
        thr[j] = best_t
        chosen += 1

    info = {"per_label_min_pos": int(min_pos), "n_labels": int(n_labels), "tuned": int(chosen), "skipped": int(skipped)}
    return thr, info


def apply_thresholds(y_score: np.ndarray, thr: float | np.ndarray) -> np.ndarray:
    if np.isscalar(thr):
        return (y_score >= float(thr)).astype(np.int8)
    else:
        thr = np.asarray(thr).reshape(1, -1)
        return (y_score >= thr).astype(np.int8)


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict:
    # F1
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUPRC (micro)
    # average_precision_score supports multilabel indicator; with average="micro" it flattens all labels.
    auprc_micro = average_precision_score(y_true, y_score, average="micro")

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "auprc_micro": float(auprc_micro),
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
    }


# ----------------------------
# Bundle for saving
# ----------------------------
@dataclass
class ModelBundle:
    vectorizer: TfidfVectorizer
    classifier: OneVsRestClassifier
    binarizer: MultiLabelBinarizer
    threshold_strategy: str
    threshold: object  # float or np.ndarray
    params: Dict
    label_list: List[str]


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to MF_dataset.parquet")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for artifacts")

    # Split
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # k-mer TF-IDF
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--min_df", type=float, default=2.0, help="int or float; passed to TfidfVectorizer")
    p.add_argument("--max_features", type=int, default=200_000)
    p.add_argument("--sublinear_tf", action="store_true", help="Use sublinear tf scaling")

    # LR
    p.add_argument("--C", type=float, default=2.0)
    p.add_argument("--solver", type=str, default="liblinear", choices=["liblinear", "saga"])
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--n_jobs", type=int, default=-1)

    # Thresholding
    p.add_argument("--threshold_strategy", type=str, default="global",
                   choices=["global", "per_label", "fixed"])
    p.add_argument("--fixed_threshold", type=float, default=0.2)
    p.add_argument("--thr_grid_min", type=float, default=0.05)
    p.add_argument("--thr_grid_max", type=float, default=0.50)
    p.add_argument("--thr_grid_steps", type=int, default=46, help="Number of thresholds in grid")
    p.add_argument("--per_label_min_pos", type=int, default=5)

    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_parquet(args.data)
    print("After load/clean, n_total =", len(df))
    assert len(df) > 0, "Dataset is empty after loading/cleaning."

    print("df shape:", df.shape)
    print("df columns:", df.columns.tolist())
    print("labels head:", df["labels"].head(5).tolist())
    print("nonempty ratio:", df["labels"].apply(len).gt(0).mean())
    print("n_labels head:", df["n_labels"].head(5).tolist())


    needed = {"accession", "sequence", "labels", "n_labels"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    # Basic cleanup (robustness)
    df = df.dropna(subset=["sequence", "labels"]).copy()
    df["sequence"] = df["sequence"].astype(str)
    df["labels"] = df["labels"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else list(x))
    print("BEFORE Mlb fit_transform labels head:", df["labels"].head(5).tolist())
    print("BEFORE nonempty ratio:", df["labels"].apply(len).gt(0).mean())
    # Binarize labels on full set (consistent label space), but you'll fit model on train only.
    mlb = MultiLabelBinarizer(sparse_output=False)
   
    
    
    Y_all = mlb.fit_transform(df["labels"])


    print("labels col dtype:", df["labels"].dtype)
    print("labels sample raw:", df["labels"].head(5).tolist())
    nonempty = df["labels"].apply(lambda x: isinstance(x, (list, tuple, set)) and len(x) > 0).mean()
    print("nonempty label ratio:", nonempty)
  
    print("len(mlb.classes_):", len(mlb.classes_))
    print("classes head:", list(mlb.classes_[:10]) if len(mlb.classes_) else mlb.classes_)
    print("Y_all shape:", Y_all.shape)

    label_list = list(mlb.classes_)

    # Split
    df_tr, df_va = train_val_split(df, val_ratio=args.val_ratio, seed=args.seed)
    y_tr = df_tr["labels"].tolist()
    y_va = df_va["labels"].tolist()
    Y_tr = mlb.transform(y_tr)
    Y_va = mlb.transform(y_va)
    print("Y_tr shape:", Y_tr.shape)


    split_summary = summarize_split(df_tr, df_va, Y_tr, Y_va)
    min_df = args.min_df
    if isinstance(min_df, float) and min_df > 1.0:
        min_df = int(min_df)  # 2.0 -
    print("val_ratio =", args.val_ratio)
    print("After split: n_train =", len(df_tr), "n_val =", len(df_va))
    assert len(df_tr) > 0, "Train split is empty. Check val_ratio or filtering."
    assert len(df_va) > 0, "Val split is empty. Check val_ratio."
    # Vectorizer
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(args.k, args.k),
        lowercase=False,          # protein sequences are uppercase
        min_df=min_df,
        max_features=args.max_features,
        sublinear_tf=args.sublinear_tf,
        norm="l2",
        dtype=np.float32,
    )

    # --- DEBUG: check sequences and params ---
    seqs = df_tr["sequence"].astype(str).fillna("")
    print("k =", args.k, "min_df =", min_df, type(min_df), "max_features =", args.max_features)
    print("n_train =", len(seqs))
    print("empty seq ratio =", float((seqs.str.len() == 0).mean()))
    print("min/median/max seq len =", int(seqs.str.len().min()), int(seqs.str.len().median()), int(seqs.str.len().max()))

# quick k-mer existence sanity: how many seqs have length >= k?
    print("ratio len>=k =", float((seqs.str.len() >= args.k).mean()))
# ----------------------------------------



    X_tr = vectorizer.fit_transform(df_tr["sequence"])
    X_va = vectorizer.transform(df_va["sequence"])

    # Classifier
    base_lr = LogisticRegression(
        C=args.C,
        solver="liblinear", 
        max_iter=args.max_iter,
        n_jobs=args.n_jobs if args.solver in ("saga",) else None,  # liblinear ignores n_jobs
        random_state=args.seed,
    )
    clf = OneVsRestClassifier(base_lr, n_jobs=1)

    clf.fit(X_tr, Y_tr)

    # Scores on val
    # OneVsRestClassifier with LogisticRegression provides decision_function and predict_proba depending on solver.
    if hasattr(clf, "predict_proba"):
        Y_va_score = clf.predict_proba(X_va)
    else:
        # fallback: sigmoid(decision_function) if needed (rare here)
        scores = clf.decision_function(X_va)
        Y_va_score = 1.0 / (1.0 + np.exp(-scores))

    # Threshold selection
    grid = np.linspace(args.thr_grid_min, args.thr_grid_max, args.thr_grid_steps, dtype=np.float32)

    thr_info: Dict = {}
    if args.threshold_strategy == "fixed":
        thr = float(args.fixed_threshold)
        thr_info = {"strategy": "fixed", "threshold": thr}
    elif args.threshold_strategy == "global":
        thr, info = select_global_threshold_by_micro_f1(Y_va, Y_va_score, grid)
        thr_info = {"strategy": "global", "threshold": float(thr), **info}
    elif args.threshold_strategy == "per_label":
        thr, info = select_per_label_thresholds_by_f1(
            Y_va, Y_va_score, grid, min_pos=args.per_label_min_pos
        )
        thr_info = {"strategy": "per_label", **info}
    else:
        raise ValueError(f"Unknown threshold_strategy: {args.threshold_strategy}")

    Y_va_pred = apply_thresholds(Y_va_score, thr)

    # Metrics
    metrics = compute_metrics(Y_va, Y_va_score, Y_va_pred)

    # Save artifacts
    params = vars(args)
    bundle = ModelBundle(
        vectorizer=vectorizer,
        classifier=clf,
        binarizer=mlb,
        threshold_strategy=args.threshold_strategy,
        threshold=thr,
        params=params,
        label_list=label_list,
    )
    joblib_path = outdir / "model.joblib"
    joblib.dump(bundle, joblib_path)

    report = {
        "data": str(args.data),
        "outdir": str(outdir),
        "split_summary": split_summary,
        "val_metrics": metrics,
        "threshold_info": thr_info,
        "feature_space": {
            "k": args.k,
            "min_df": args.min_df,
            "max_features": args.max_features,
            "n_features_actual": int(getattr(vectorizer, "vocabulary_", {}) and len(vectorizer.vocabulary_) or 0),
        },
        "lr": {
            "C": args.C,
            "solver": args.solver,
            "max_iter": args.max_iter,
        },
    }
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary to stdout (nice for quick check)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
