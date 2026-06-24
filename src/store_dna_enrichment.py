"""
Stage 3 — AI Enrichment for Retail StoreDNA.

Reads curated CSVs, builds embeddable text per row, calls Azure OpenAI
embeddings (and optional GPT enrichment), uploads vectors to Azure AI Search,
writes enriched metadata CSVs locally.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

from src.store_dna_search import (
    build_search_documents,
    ensure_embedding_index,
    upload_embedding_documents,
)

MODALITIES = ("reviews", "news", "reports", "products", "ops_weekly")


def _embed_field(row: pd.Series, label: str, column: str) -> str | None:
    """Format one labeled field for embed_text; skip missing/blank values."""
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value) or str(value).strip() == "":
        return None
    if label:
        return f"{label}: {value}"
    return str(value)


def _embed_from_fields(row: pd.Series, fields: list[tuple[str, str]]) -> str:
    parts = [part for label, col in fields if (part := _embed_field(row, label, col))]
    return " | ".join(parts) if parts else "empty"


@dataclass
class EnrichmentConfig:
    """Azure OpenAI settings loaded from environment / .env."""

    api_key: str
    endpoint: str
    api_version: str
    gpt_deployment: str
    embedding_deployment: str
    embed_batch_size: int = 50
    gpt_temperature: float = 0.2
    run_gpt_enrichment: bool = True
    max_stores: int | None = None  # limit stores for dev runs
    search_endpoint: str = ""
    search_api_key: str = ""
    search_index_name: str = "store-dna-embeddings"
    embedding_dimensions: int = 3072
    search_upload_batch_size: int = 500

    @classmethod
    def from_env(cls, project_root: Path | None = None, **overrides: Any) -> EnrichmentConfig:
        if project_root:
            load_dotenv(project_root / ".env")
        return cls(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            gpt_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt4-8k"),
            embedding_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/"),
            search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
            search_index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "store-dna-embeddings"),
            embedding_dimensions=int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "3072")),
            search_upload_batch_size=int(os.getenv("AZURE_SEARCH_UPLOAD_BATCH_SIZE", "500")),
            **overrides,
        )

    def create_client(self) -> AzureOpenAI:
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )

    def search_config(self):
        from src.store_dna_search import SearchIndexConfig

        return SearchIndexConfig(
            endpoint=self.search_endpoint,
            api_key=self.search_api_key,
            index_name=self.search_index_name,
            vector_dimensions=self.embedding_dimensions,
            upload_batch_size=self.search_upload_batch_size,
        )


def load_curated_tables(curated_dir: Path) -> dict[str, pd.DataFrame]:
    """Load Stage 2 curated outputs."""
    return {
        "dim_store": pd.read_csv(curated_dir / "dim_store.csv"),
        "reviews": pd.read_csv(curated_dir / "stg_reviews.csv"),
        "news": pd.read_csv(curated_dir / "stg_news.csv"),
        "reports": pd.read_csv(curated_dir / "stg_reports.csv"),
        "products": pd.read_csv(curated_dir / "stg_products.csv"),
        "ops": pd.read_csv(curated_dir / "fact_operations_weekly.csv", parse_dates=["week_end_date"]),
    }


def filter_stores(tables: dict[str, pd.DataFrame], store_ids: set[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for name, df in tables.items():
        if name == "dim_store":
            out[name] = df[df["store_id"].isin(store_ids)].copy()
        elif "store_id" in df.columns:
            out[name] = df[df["store_id"].isin(store_ids)].copy()
        else:
            out[name] = df.copy()
    return out


def build_review_embed_text(row: pd.Series) -> str:
    parts = [f"Store {row['store_id']}", f"Source: {row['source']}"]
    if pd.notna(row.get("sentiment")) and str(row["sentiment"]).strip():
        parts.append(f"Sentiment: {row['sentiment']}")
    parts.append(str(row["review_text"]))
    return " | ".join(parts)


def build_news_embed_text(row: pd.Series) -> str:
    return (
        f"Store {row['store_id']} | {row.get('city', '')}, {row.get('state', '')} | "
        f"Date: {row.get('published_date', '')} | "
        f"Headline: {row['headline']} | Summary: {row.get('summary', '')} | "
        f"Impact: {row.get('demand_impact', '')}"
    )


def build_report_embed_text(row: pd.Series) -> str:
    return _embed_from_fields(row, [
        ("Store", "store_id"),
        ("Report", "report_id"),
        ("Date", "report_date"),
        ("Type", "report_type"),
        ("Severity", "severity"),
        ("Category", "issue_category"),
        ("Status", "status"),
        ("Reported by", "reported_by"),
        ("Description", "description"),
    ])


def build_product_embed_text(row: pd.Series) -> str:
    return _embed_from_fields(row, [
        ("Store", "store_id"),
        ("SKU", "sku_id"),
        ("Category", "category"),
        ("Product", "product_name"),
        ("Description", "description"),
        ("Attributes", "attributes"),
        ("Brand", "brand_type"),
        ("Price tier", "price_tier"),
        ("Unit price", "unit_price"),
        ("Pack size", "pack_size"),
        ("Aisle", "aisle_zone"),
        ("Facings", "facings"),
        ("Weekly units sold", "weekly_units_sold"),
        ("Margin pct", "margin_pct"),
        ("In stock", "in_stock"),
        ("Vendor", "vendor"),
    ])


def build_ops_weekly_embed_text(row: pd.Series) -> str:
    return _embed_from_fields(row, [
        ("Store", "store_id"),
        ("Week ending", "week_end_date"),
        ("Labor hours", "labor_hours"),
        ("Shrink pct", "shrink_pct"),
        ("OOS rate", "oos_rate"),
        ("Fulfillment rate", "fulfillment_rate"),
        ("Customer complaints", "customer_complaints"),
    ])


def build_ops_store_summary(store_id: str, ops_df: pd.DataFrame) -> str:
    """Aggregate weekly ops into one narrative block per store for embedding."""
    sub = ops_df[ops_df["store_id"] == store_id]
    if sub.empty:
        return f"Store {store_id} | No operations data."
    agg = sub.agg({
        "labor_hours": "mean",
        "shrink_pct": "mean",
        "oos_rate": "mean",
        "fulfillment_rate": "mean",
        "customer_complaints": "sum",
    })
    return (
        f"Store {store_id} operations profile ({len(sub)} weeks): "
        f"avg labor hours {agg['labor_hours']:.0f}, "
        f"shrink {agg['shrink_pct']:.2%}, "
        f"out-of-stock rate {agg['oos_rate']:.2%}, "
        f"fulfillment {agg['fulfillment_rate']:.1%}, "
        f"total complaints {int(agg['customer_complaints'])}."
    )


EMBED_TEXT_BUILDERS = {
    "reviews": build_review_embed_text,
    "news": build_news_embed_text,
    "reports": build_report_embed_text,
    "products": build_product_embed_text,
    "ops_weekly": build_ops_weekly_embed_text,
}

ID_COLUMNS = {
    "reviews": "review_id",
    "news": "news_id",
    "reports": "report_id",
    "products": "sku_id",
    "ops_weekly": "ops_row_id",
}


def add_embed_text(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    out = df.copy()
    builder = EMBED_TEXT_BUILDERS[modality]
    out["embed_text"] = out.apply(builder, axis=1)
    return out


def embed_texts(
    client: AzureOpenAI,
    texts: list[str],
    deployment: str,
    batch_size: int = 50,
    pause_sec: float = 0.5,
    progress_label: str | None = None,
) -> np.ndarray:
    """Return (n, dim) embedding matrix for a list of texts."""
    vectors: list[list[float]] = []
    clean = [t if isinstance(t, str) and t.strip() else "empty" for t in texts]
    total = len(clean)

    for start in range(0, len(clean), batch_size):
        batch = clean[start : start + batch_size]
        if progress_label:
            end = min(start + len(batch), total)
            print(f"[Stage 3] {progress_label}: embedding rows {start + 1:,}-{end:,} of {total:,}")
        response = client.embeddings.create(input=batch, model=deployment)
        ordered = sorted(response.data, key=lambda x: x.index)
        vectors.extend([item.embedding for item in ordered])
        if start + batch_size < len(clean):
            time.sleep(pause_sec)

    return np.array(vectors, dtype=np.float32)


def gpt_store_modality_summary(
    client: AzureOpenAI,
    deployment: str,
    store_id: str,
    retailer: str,
    modality: str,
    sample_texts: list[str],
    temperature: float = 0.2,
) -> dict[str, str]:
    """One GPT call per store × modality — concise summary for downstream use."""
    joined = "\n".join(f"- {t[:400]}" for t in sample_texts[:12])
    prompt = f"""You are building Retail StoreDNA enrichment.

Store: {store_id} ({retailer})
Modality: {modality}

Summarize the following items into JSON with exactly these keys:
- "summary": 2-3 sentences capturing store-level signal for this modality
- "themes": comma-separated short theme tags (max 8)
- "sentiment_overall": one of positive, neutral, negative, mixed

Items:
{joined}

Return ONLY valid JSON."""

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content or "{}"
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"summary": content[:500], "themes": "", "sentiment_overall": "neutral"}
    return {
        "ai_summary": str(parsed.get("summary", "")),
        "ai_themes": str(parsed.get("themes", "")),
        "ai_sentiment_overall": str(parsed.get("sentiment_overall", "neutral")),
    }


def enrich_modality(
    client: AzureOpenAI,
    config: EnrichmentConfig,
    modality: str,
    df: pd.DataFrame,
    dim_store: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Add embed_text, call embeddings API, optional GPT store summaries."""
    enriched = add_embed_text(df, modality)

    embeddings = embed_texts(
        client,
        enriched["embed_text"].tolist(),
        config.embedding_deployment,
        batch_size=config.embed_batch_size,
        progress_label=modality,
    )

    enriched["embedding_model"] = config.embedding_deployment
    enriched["embedding_dim"] = embeddings.shape[1] if len(embeddings) else 0

    if config.run_gpt_enrichment and modality != "ops_weekly":
        summaries = []
        retailer_map = dim_store.set_index("store_id")["retailer"].to_dict()
        for store_id, grp in enriched.groupby("store_id"):
            texts = grp["embed_text"].tolist()
            meta = gpt_store_modality_summary(
                client,
                config.gpt_deployment,
                store_id,
                retailer_map.get(store_id, ""),
                modality,
                texts,
                config.gpt_temperature,
            )
            meta["store_id"] = store_id
            meta["modality"] = modality
            summaries.append(meta)
            time.sleep(0.3)
        summary_df = pd.DataFrame(summaries)
        enriched = enriched.merge(summary_df, on="store_id", how="left")

    return enriched, embeddings


def run_enrichment(
    curated_dir: Path,
    output_dir: Path,
    config: EnrichmentConfig,
) -> dict[str, Any]:
    """Full Stage 3 pipeline — embeddings uploaded to Azure AI Search."""
    start_time = time.time()
    print("[Stage 3] Loading curated tables...")
    tables = load_curated_tables(curated_dir)
    if config.max_stores:
        store_ids = set(tables["dim_store"]["store_id"].head(config.max_stores))
        tables = filter_stores(tables, store_ids)
        print(f"[Stage 3] Store filter applied: {len(store_ids)} stores")
    else:
        print(f"[Stage 3] Processing all stores: {tables['dim_store']['store_id'].nunique()}")

    print("[Stage 3] Creating Azure OpenAI client...")
    client = config.create_client()
    search_cfg = config.search_config()
    print(f"[Stage 3] Ensuring Azure AI Search index: {config.search_index_name}")
    ensure_embedding_index(search_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    retailer_map = tables["dim_store"].set_index("store_id")["retailer"].to_dict()

    result: dict[str, Any] = {
        "search_index": config.search_index_name,
        "modalities": {},
    }

    # Row-level modalities — every curated row embedded (including each store × week)
    ops_df = tables["ops"].copy()
    if not ops_df.empty:
        ops_df["week_end_date"] = pd.to_datetime(ops_df["week_end_date"]).dt.strftime("%Y-%m-%d")
        ops_df["ops_row_id"] = ops_df["store_id"].astype(str) + "_" + ops_df["week_end_date"]
        tables["ops_weekly"] = ops_df

    for modality in ("reviews", "news", "reports", "products", "ops_weekly"):
        df = tables.get(modality)
        if df is None or df.empty:
            print(f"[Stage 3] Skipping {modality}: no rows")
            continue
        modality_start = time.time()
        print(f"[Stage 3] {modality}: {len(df):,} rows -> building embed_text")
        enriched, emb = enrich_modality(client, config, modality, df, tables["dim_store"])
        out_name = "ops_weekly" if modality == "ops_weekly" else modality
        print(f"[Stage 3] {modality}: embeddings ready with shape {emb.shape}")
        embed_texts = enriched["embed_text"].tolist()
        documents = build_search_documents(
            modality,
            enriched,
            emb,
            ID_COLUMNS[modality],
            embed_texts,
            retailer_map,
            config.embedding_deployment,
        )
        print(f"[Stage 3] {modality}: uploading {len(documents):,} docs to Azure AI Search")
        indexed = upload_embedding_documents(search_cfg, documents)
        enriched.drop(columns=["embed_text"], errors="ignore").to_csv(
            output_dir / f"{out_name}_enriched.csv", index=False
        )
        elapsed = time.time() - modality_start
        print(
            f"[Stage 3] {modality}: wrote {out_name}_enriched.csv, "
            f"indexed {indexed:,} docs in {elapsed:.1f}s"
        )
        result["modalities"][modality] = {
            "rows": len(enriched),
            "indexed": indexed,
        }

    # Store-level GPT summaries file (from merged columns on enriched tables)
    print("[Stage 3] Building store_modality_summaries.csv...")
    store_summaries = []
    for modality in ("reviews", "news", "reports", "products"):
        path = output_dir / f"{modality}_enriched.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "ai_summary" not in df.columns:
            continue
        sub = df[["store_id", "ai_summary", "ai_themes", "ai_sentiment_overall"]].drop_duplicates("store_id")
        sub["modality"] = modality
        store_summaries.append(sub)
    if store_summaries:
        pd.concat(store_summaries, ignore_index=True).to_csv(
            output_dir / "store_modality_summaries.csv", index=False
        )
        print("[Stage 3] Wrote store_modality_summaries.csv")

    total_elapsed = time.time() - start_time
    print(f"[Stage 3] Complete in {total_elapsed:.1f}s")
    return result
