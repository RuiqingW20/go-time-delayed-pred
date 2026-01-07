python - <<'EOF'
from pathlib import Path
from src.encoding.io import parse_fasta, read_train_terms

train_proteins = parse_fasta(Path("data/raw/train_sequences.fasta"))
terms = read_train_terms(Path("data/raw/train_terms.tsv"))

s_prot = set(train_proteins["accession"])
s_term = set(terms["accession"])

print("n_proteins:", len(s_prot))
print("n_term_accessions:", len(s_term))
print("intersection:", len(s_prot & s_term))

print("train_proteins accession examples:", train_proteins["accession"].head(5).tolist())
print("terms accession examples:", terms["accession"].head(5).tolist())
EOF
