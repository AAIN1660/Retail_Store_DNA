# Retail Store DNA — Front end

Same three pages via FastAPI:

| Tab | Source |
|-----|--------|
| Executive Summary / KPIs | Curated CSVs |
| Store Pulse themes & summaries | Embeddings + Azure OpenAI LLM (cached) |
| Store Pulse local news | Azure OpenAI headlines + summaries (cached) |
| Similar stores | Stage 6 Azure store embeddings |

Review insights cache: `data/USA_100_Stores/enriched/review_pulse_insights.json`  
News insights cache: `data/USA_100_Stores/enriched/news_pulse_insights.json`  
First load generates missing stores (can take several minutes). Force refresh:

```bash
# PowerShell
$env:REVIEW_INSIGHTS_FORCE_REFRESH='1'
$env:NEWS_INSIGHTS_FORCE_REFRESH='1'
python -m uvicorn api.main:app --reload --port 8000
```

Optional limits while testing: `REVIEW_INSIGHTS_MAX_STORES` / `NEWS_INSIGHTS_MAX_STORES`.

## Prerequisites

1. Curated data present: `data/USA_100_Stores/curated/` (and enriched reviews/news when available).
2. Stage 6 index populated: `notebooks` Stage 6 / vector index → Azure `store-dna-store-vectors`.
3. `.env` includes Azure Search endpoint/key and `AZURE_SEARCH_STORE_INDEX_NAME=store-dna-store-vectors`.

## Run

**Terminal 1 — API** (project root):

```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Front end:**

```bash
cd "Front end"
npm install
npm run dev
```

Open the Vite URL (usually http://localhost:5173).
