# Retail Store DNA

Store-level vector embeddings from structured and unstructured retail data — scrape, curate, enrich, and build StoreDNA fingerprints for 100 US retail stores.

## Pipeline (local dev)

1. **Source Data** — Excel workbooks (`data/USA_100_Stores/`)
2. **Curate Data** — `notebooks/Retail_Store_DNA_Builder_Stage_2_Data_Curation.ipynb`
3. **AI Enrichment** — Azure OpenAI (embeddings + GPT)
4. **Modality Vectors** — per-signal embeddings
5. **StoreDNA Builder** — late fusion
6. **Vector Index** — Azure AI Search
7. **Business Output** — peers, clusters, narratives

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add Azure OpenAI keys locally
```

## Key paths

| Path | Description |
|------|-------------|
| `data/USA_100_Stores/` | Source Excel (scraped reviews/news + synthetic ops/products) |
| `data/USA_100_Stores/curated/` | Stage 2 curated CSV outputs |
| `notebooks/04_scrape_reviews_news_100_stores.ipynb` | Scrape 100-store reviews & news |
| `notebooks/Retail_Store_DNA_Builder_Stage_2_Data_Curation.ipynb` | Curate data for StoreDNA |
| `src/store_dna_curator.py` | Curation helpers |

## Branch

Primary development branch: **Chandana**
