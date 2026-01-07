from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

_TAXON_PATTERNS = [
    re.compile(r"taxon[:=](\d+)"),  # e.g. taxon:9606 or taxon=9606
    re.compile(r"OX=(\d+)"),        # UniProt style
]


def parse_fasta(path: Path) -> pd.DataFrame:
    """
    Parse FASTA into a DataFrame with columns:
      - accession (str): first token in header
      - sequence (str)
      - taxon_id (int | None): parsed if available, else None
    """
    records: list[dict] = []
    header: str | None = None
    seq_parts: list[str] = []

    def flush():
        nonlocal header, seq_parts
        if header is None:
            return
        seq = "".join(seq_parts).replace(" ", "").strip()
        if not seq:
            return

        token = header.split()[0].strip()
        if token.startswith(("sp|", "tr|")) and token.count("|") >= 2:
            acc = token.split("|", 2)[1].strip()
        else:
    # fallback: if still contains pipes, take the 2nd field if possible
            if "|" in token:
                parts = token.split("|")
                acc = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else token
            else:
                acc = token
        taxon_id = None
        for pat in _TAXON_PATTERNS:
            m = pat.search(header)
            if m:
                taxon_id = int(m.group(1))
                break

        records.append({"accession": acc, "sequence": seq, "taxon_id": taxon_id})

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        flush()

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # De-dup by accession (keep first occurrence)
    df = df.drop_duplicates(subset=["accession"]).reset_index(drop=True)
    return df


def read_train_terms(path: Path) -> pd.DataFrame:
    """
    train_terms.tsv columns (raw):
      - EntryID : protein accession
      - term    : GO term ID (e.g. GO:0008150)
      - aspect  : ontology (MF / BP / CC)

    Internal canonical columns:
      - accession
      - go_id
      - aspect
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["EntryID", "term", "aspect"],
    )

    # basic cleanup
    df = df.dropna()

    # rename to canonical schema
    df = df.rename(
        columns={
            "EntryID": "accession",
            "term": "go_id",
        }
    )

    df["accession"] = df["accession"].astype(str).str.strip()
    df["go_id"] = df["go_id"].astype(str).str.strip()
    df["aspect"] = df["aspect"].astype(str).str.strip().str.upper()

    df["aspect"] = df["aspect"].replace({"F": "MF", "P": "BP", "C": "CC"})
    df = df[df["go_id"].str.startswith("GO:")]


    # remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df