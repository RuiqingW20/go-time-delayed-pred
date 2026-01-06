from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import pandas as pd


def filter_terms_by_freq(train_terms: pd.DataFrame, aspect: str, min_freq: int) -> tuple[set[str], dict[str, int]]:
    """
    Returns (kept_terms, term_counts_for_aspect)
    """
    df = train_terms[train_terms["aspect"] == aspect]
    cnt = Counter(df["go_id"].tolist())
    kept = {go for go, c in cnt.items() if c >= min_freq}
    return kept, dict(cnt)


def aggregate_labels(train_terms: pd.DataFrame, aspect: str, kept_terms: set[str]) -> pd.DataFrame:
    """
    Output columns: accession, labels(list[str]), n_labels(int)
    """
    ann = train_terms[train_terms["aspect"] == aspect][["accession", "go_id"]].drop_duplicates()
    ann = ann[ann["go_id"].isin(kept_terms)]

    labels = (
        ann.groupby("accession")["go_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"go_id": "labels"})
    )
    labels["n_labels"] = labels["labels"].map(len)
    return labels


def save_label_vocab(
    out_path: Path,
    aspect: str,
    min_freq: int,
    kept_terms: set[str],
    term_counts: dict[str, int],
) -> None:
    """
    Save vocab JSON for reproducibility.
    """
    obj = {
        "aspect": aspect,
        "min_term_freq": int(min_freq),
        "n_terms_kept": int(len(kept_terms)),
        "go_terms": sorted(list(kept_terms)),
        "term_counts": term_counts,  # raw counts in this aspect
    }
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
