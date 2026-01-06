python - << 'EOF'
import pandas as pd

terms = pd.read_csv("data/raw/train_terms.tsv", sep="\t")
mf = terms[terms["aspect"].astype(str).str.upper().isin(["MF","F"])]
print("MF rows:", len(mf))
if len(mf)==0:
    print("No MF rows after filtering. Check aspect encoding (MF vs F) and column name.")
else:
    vc = mf["EntryID"].value_counts()
    print("unique MF terms:", vc.size)
    for k in [1,2,5,10,20]:
        print(f"terms with freq >= {k}:", int((vc>=k).sum()))
    print("top 10 freqs:", vc.head(10).to_dict())
p="data/raw/train_terms.tsv"
df1=pd.read_csv(p, sep="\t", nrows=5, dtype=str)
print("read with header columns:", df1.columns.tolist())
print(df1.to_string(index=False))

df2=pd.read_csv(p, sep="\t", header=None, names=["EntryID","term","aspect"], nrows=5, dtype=str)
print("\nread as headerless:")
print(df2.to_string(index=False))

# check aspect weirdness
df=pd.read_csv(p, sep="\t", dtype=str)
col = "aspect" if "aspect" in df.columns else (df.columns[2] if len(df.columns)>=3 else None)
if col:
    s=df[col].astype(str)
    print("\nraw aspect top:", s.value_counts().head(10).to_dict())
    print("after strip+upper top:", s.str.strip().str.upper().value_counts().head(10).to_dict())
EOF






