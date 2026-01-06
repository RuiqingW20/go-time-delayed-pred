We use the CAFA-style GO annotation dataset with the following raw files:

train_sequences.fasta: protein sequences with UniProt accessions.

train_terms.tsv: protein–GO annotations with ontology labels.

testsuperset.fasta: unlabeled protein sequences for downstream prediction.

IA.tsv: information accretion weights for GO evaluation (used later).

The raw annotation file train_terms.tsv contains three columns with header:

EntryID   term          aspect
Q5W0B1    GO:0000785    C
Q5W0B1    GO:0004842    F
...

where aspect ∈ {F, P, C} denotes Molecular Function (MF), Biological Process (BP), and Cellular Component (CC), respectively.

2. Canonical schema and normalization

To ensure consistency across the pipeline, we normalize the annotation table into a canonical internal schema:

EntryID → accession

term → go_id

ontology codes are mapped as F → MF, P → BP, C → CC

whitespace is stripped and aspect codes are uppercased

rows with invalid GO IDs (not starting with GO:) are removed

exact duplicates are dropped

3. Sequence parsing

Protein sequences are parsed from FASTA files using accession identifiers extracted from the header line.
For each protein, we store:

accession

sequence

taxon_id (parsed when available, otherwise None)

Parsed protein tables are saved as intermediate artifacts:

data/interim/train_proteins.parquet

data/interim/test_proteins.parquet

Duplicate accessions are removed by keeping the first occurrence.


4. Label frequency analysis and filtering

GO annotations exhibit an extreme long-tail distribution, particularly for Molecular Function (MF).
We compute per-aspect GO term frequencies on the training annotations and apply a minimum frequency threshold:

min_term_freq ∈ {5, 10, 20}


For MF, the observed distribution is:

threshold	# GO terms
≥ 1	58,001
≥ 2	30,119
≥ 5	5,315
≥ 10	614
≥ 20	35

Based on this distribution, we select:

min_term_freq = 10 for a stable baseline,

lower thresholds (e.g. 5) for more realistic and challenging settings.

Terms below the chosen threshold are excluded from the label space.
