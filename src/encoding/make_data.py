from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.encoding.io import parse_fasta, read_train_terms
from src.encoding.make_labels import filter_terms_by_freq, aggregate_labels, save_label_vocab


def main():
    p = argparse.ArgumentParser(description="Preprocess FASTA and build multilabel dataset (sequence-only).")
    p.add_argument("--data_dir", type=str, default="data", help="Base data dir containing raw/")
    p.add_argument("--aspect", type=str, default="MF", choices=["MF", "BP", "CC"], help="Ontology/aspect to build labels for.")
    p.add_argument("--min_term_freq", type=int, default=20, help="Keep GO terms with >= this frequency.")
    p.add_argument("--min_labels_per_protein", type=int, default=1, help="Filter proteins with fewer labels.")
    p.add_argument("--max_labels_per_protein", type=int, default=50, help="Filter proteins with more labels.")
    args = p.parse_args()

    base = Path(args.data_dir)
    raw = base / "raw"
    interim = base / "interim"
    processed = base / "processed"
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    # 1) Read FASTA
    train_proteins_path = raw / "train_sequences.fasta"
    test_proteins_path = raw / "testsuperset.fasta"

    train_proteins = parse_fasta(train_proteins_path)
    test_proteins = parse_fasta(test_proteins_path)

    # Save interim proteins
    train_proteins.to_parquet(interim / "train_proteins.parquet", index=False)
    test_proteins.to_parquet(interim / "test_proteins.parquet", index=False)

    if train_proteins.empty:
        raise RuntimeError("Parsed train_sequences.fasta but got 0 proteins. Check file format/path.")

    # 2) Read train_terms.tsv
    terms_path = raw / "train_terms.tsv"
    train_terms = read_train_terms(terms_path)

    # Basic alignment sanity check
    n_terms_total = len(train_terms)
    n_terms_in_train_fasta = train_terms["accession"].isin(set(train_proteins["accession"])).sum()

    # 3) Build kept terms (label vocab)
    kept_terms, term_counts = filter_terms_by_freq(train_terms, args.aspect, args.min_term_freq)
    if len(kept_terms) == 0:
        raise RuntimeError(f"No GO terms kept for aspect={args.aspect} with min_term_freq={args.min_term_freq}")

    vocab_path = processed / f"{args.aspect}_label_vocab.json"
    save_label_vocab(vocab_path, args.aspect, args.min_term_freq, kept_terms, term_counts)

    # 4) Aggregate labels per protein and join with sequences
    labels = aggregate_labels(train_terms, args.aspect, kept_terms)
    df = train_proteins.merge(labels, on="accession", how="inner")

    # 5) Protein-level filtering by label count
    df["n_labels"] = df["labels"].map(len)
    df = df[(df["n_labels"] >= args.min_labels_per_protein) & (df["n_labels"] <= args.max_labels_per_protein)].copy()
    df = df.reset_index(drop=True)

    out_dataset = processed / f"{args.aspect}_dataset.parquet"
    df.to_parquet(out_dataset, index=False)

    # 6) Print summary
    train_label_set = set(t for labs in df["labels"] for t in labs)
    print("==== Preprocess summary ====")
    print(f"Train proteins parsed: {len(train_proteins):,}")
    print(f"Test proteins parsed : {len(test_proteins):,}")
    print(f"Train terms rows     : {n_terms_total:,}")
    print(f"Terms accessions in train fasta: {n_terms_in_train_fasta:,} ({n_terms_in_train_fasta/n_terms_total:.2%})")
    print(f"Aspect              : {args.aspect}")
    print(f"min_term_freq       : {args.min_term_freq}")
    print(f"Kept GO terms       : {len(kept_terms):,}")
    print(f"Proteins w/ labels  : {len(df):,}")
    print(f"Unique labels in df : {len(train_label_set):,}")
    print(f"Saved interim train proteins -> {interim/'train_proteins.parquet'}")
    print(f"Saved interim test proteins  -> {interim/'test_proteins.parquet'}")
    print(f"Saved vocab                 -> {vocab_path}")
    print(f"Saved dataset               -> {out_dataset}")


if __name__ == "__main__":
    main()
