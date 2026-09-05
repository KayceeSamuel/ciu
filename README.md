# BioSearch

Scientific database search library for mitochondrial biology
and clinical genetics research.

## Databases

| Database | What it provides | License |
|---|---|---|
| PubMed | 35M paper abstracts | Free |
| Europe PMC | Open access full text | Free |
| bioRxiv | Latest preprints | Free |
| UniProt | Protein annotations | Free |
| ClinVar | Variant classifications | Free |
| MITOMAP | mtDNA variants | Free |
| AlphaFold | Protein structures | Free |
| Ensembl | Gene info + VEP | Free |
| Orphanet | Rare disease clinical | Free |
| Open Targets | Gene-disease scores | Free |

## Installation

```bash
pip install -e .
# or
pip install requests beautifulsoup4
```

## Usage

```python
from biosearch import BioSearch

# Basic usage
search = BioSearch()
results = search.pubmed("POLG p.Ala467Thr Alpers syndrome")
results = search.clinvar("POLG Ala467Thr")
results = search.alphafold("POLG")

# Auto-routing — detects query type and searches appropriate APIs
results = search.query("POLG p.Ala467Thr compound heterozygous")

# With RAG injection
search = BioSearch(
    rag_corpus=my_corpus,
    rebuild_fn=build_rag_index
)
n_chunks = search.query_into_rag("ANKZF1 Complex IV deficiency")

# Specific APIs only
results = search.query(
    "POLG Alpers syndrome liver",
    apis=["pubmed", "clinvar", "orphanet"]
)

# Health check
search.test_all_apis()
```

## Using in Colab

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/MitoAgent_v3/')
from biosearch import BioSearch
```

## Tiers

- `tier=1` — PubMed only (fast, 0.5s)
- `tier=2` — Literature + protein APIs (1.5s)
- `tier=3` — All 10 APIs (4-6s)
- `tier=None` — Auto-detect from query (default)
