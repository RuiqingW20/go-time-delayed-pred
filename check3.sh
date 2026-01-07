python - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/processed/MF_dataset.parquet")
print("shape:", df.shape)
print("n_labels describe:\n", df["n_labels"].describe())
print("n_labels value_counts head:\n", df["n_labels"].value_counts().head(10).to_dict())
print("nonempty label ratio:", df["labels"].apply(len).gt(0).mean())

##python - <<'EOF'
from pathlib import Path
from src.encoding.io import read_train_terms
terms = read_train_terms(Path("data/raw/train_terms.tsv"))
print(terms["aspect"].value_counts().head(10).to_dict())
print("example aspects:", terms["aspect"].unique()[:10])
EOF
