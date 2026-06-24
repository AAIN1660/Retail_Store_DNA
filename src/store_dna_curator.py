"""
Stage 4 — Curate Data helpers for Retail StoreDNA (100 US stores).

Loads Excel workbooks from data/USA_100_Stores, validates store_id,
normalizes schemas, and writes curated tables for downstream enrichment.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

# Sheets used for modeling / StoreDNA pipeline
SCRAPED_MODEL_SHEETS = ("store_catalog", "customer_reviews", "local_news")
SCRAPED_META_SHEETS = ("scrape_run_log",)

SYNTHETIC_MODEL_SHEETS = (
    "store_catalog",
    "operations_weekly",
    "operational_reports",
    "product_descriptions",
)
SYNTHETIC_META_SHEETS = ("data_dictionary",)

DIM_STORE_COLS = [
    "store_id", "retailer", "banner", "store_name", "city", "state", "store_format",
]

REVIEWS_COLS = [
    "review_id", "store_id", "review_date", "source", "rating",
    "sentiment", "review_text", "complaint_tags",
]

NEWS_COLS = [
    "news_id", "store_id", "city", "state", "published_date", "headline",
    "summary", "event_type", "demand_impact", "source",
]

OPS_WEEKLY_COLS = [
    "store_id", "week_end_date", "labor_hours", "shrink_pct", "oos_rate",
    "fulfillment_rate", "customer_complaints",
]

REPORTS_COLS = [
    "report_id", "store_id", "report_date", "report_type", "severity",
    "issue_category", "description", "status", "reported_by",
]

PRODUCT_COLS = [
    "sku_id", "store_id", "category", "product_name", "description",
    "attributes", "brand_type", "in_stock", "price_tier", "unit_price",
    "pack_size", "aisle_zone", "facings", "weekly_units_sold", "margin_pct",
    "upc", "vendor",
]

DENORM_COLS = ("retailer", "store_name", "city", "state", "store_format")


def load_workbook_sheets(path: Path) -> dict[str, pd.DataFrame]:
    """Load all sheets from an Excel workbook into a dict."""
    xl = pd.ExcelFile(path)
    return {name: pd.read_excel(xl, name) for name in xl.sheet_names}


def _drop_denormalized(df: pd.DataFrame) -> pd.DataFrame:
    """Remove retailer/location columns duplicated from dim_store."""
    return df.drop(columns=[c for c in DENORM_COLS if c in df.columns], errors="ignore")


def _parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.date
    return out


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[cols]


def build_dim_store(
    scraped_catalog: pd.DataFrame,
    synthetic_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Single store dimension — prefer scraped catalog; validate both match."""
    dim = scraped_catalog.copy()
    dim = dim.drop_duplicates(subset=["store_id"]).sort_values("store_id")
    dim = _ensure_columns(dim, DIM_STORE_COLS)

    syn = synthetic_catalog.drop_duplicates(subset=["store_id"]).sort_values("store_id")
    merged = dim.merge(
        syn.add_suffix("_syn"),
        left_on="store_id",
        right_on="store_id_syn",
        how="outer",
        indicator=True,
    )
    only_scraped = int((merged["_merge"] == "left_only").sum())
    only_synthetic = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]
    mismatch_cols = []
    for col in DIM_STORE_COLS:
        if col == "store_id":
            continue
        syn_col = f"{col}_syn"
        if syn_col in both.columns:
            if (both[col].astype(str) != both[syn_col].astype(str)).any():
                mismatch_cols.append(col)

    qc = {
        "store_count": len(dim),
        "only_in_scraped": only_scraped,
        "only_in_synthetic": only_synthetic,
        "column_mismatches": mismatch_cols,
    }
    return dim, qc


def _source_code(source: str) -> str:
    """Short code for review_id — e.g. sitejabber → site, google_news_reviews → goog."""
    slug = re.sub(r"[^a-z0-9]", "", str(source).lower())
    return (slug[:4] or "src").upper()


def _assign_unique_review_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign review_id per (store_id, source) sequence.

    Scrape assigns REV-{store}-001..008 per source independently, so the same
    review_id can appear on sitejabber and google_news_reviews with different text.
    """
    parts: list[pd.DataFrame] = []
    for (store_id, source), grp in df.groupby(["store_id", "source"], sort=False):
        g = grp.copy()
        code = _source_code(source)
        g["review_id"] = [
            f"REV-{store_id}-{code}-{i:03d}" for i in range(1, len(g) + 1)
        ]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def curate_reviews(df: pd.DataFrame, valid_store_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = _ensure_columns(df, REVIEWS_COLS)
    out = _parse_dates(out, ["review_date"])
    before = len(out)
    out = out[out["store_id"].isin(valid_store_ids)]
    out["review_text"] = out["review_text"].astype(str).str.strip()
    out = out[out["review_text"].str.len() >= 20]
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")

    # Only remove exact same snippet from the same source — not on review_id alone
    exact_dupes = int(out.duplicated(subset=["store_id", "source", "review_text"]).sum())
    out = out.drop_duplicates(subset=["store_id", "source", "review_text"], keep="first")

    # Fix colliding scrape IDs (same REV-USR-001-001 on different sources)
    out = _assign_unique_review_ids(out)

    qc = {
        "input_rows": before,
        "output_rows": len(out),
        "exact_duplicate_rows_removed": exact_dupes,
        "orphan_store_ids": int((~df["store_id"].isin(valid_store_ids)).sum()),
        "note": (
            "review_id reassigned per store+source; "
            "multiple rows per store kept when source or review_text differs"
        ),
    }
    return out.reset_index(drop=True), qc


def curate_news(df: pd.DataFrame, valid_store_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = _ensure_columns(df, NEWS_COLS)
    out = _parse_dates(out, ["published_date"])
    before = len(out)
    out = out.drop_duplicates(subset=["news_id"])
    out = out[out["store_id"].isin(valid_store_ids)]
    out["headline"] = out["headline"].astype(str).str.strip()
    out = out[out["headline"].str.len() >= 10]
    dup_headlines = int(out.duplicated(subset=["store_id", "headline"]).sum())
    out = out.drop_duplicates(subset=["store_id", "headline"], keep="first")
    qc = {
        "input_rows": before,
        "output_rows": len(out),
        "duplicate_headlines_removed": dup_headlines,
        "orphan_store_ids": int((~df["store_id"].isin(valid_store_ids)).sum()),
    }
    return out.reset_index(drop=True), qc


def curate_operations_weekly(df: pd.DataFrame, valid_store_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = _drop_denormalized(df)
    out = _ensure_columns(out, OPS_WEEKLY_COLS)
    out = _parse_dates(out, ["week_end_date"])
    out = out[out["store_id"].isin(valid_store_ids)]
    for col in ("shrink_pct", "oos_rate", "fulfillment_rate", "labor_hours"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["customer_complaints"] = pd.to_numeric(
        out["customer_complaints"], errors="coerce"
    ).fillna(0).astype(int)
    out = out.drop_duplicates(subset=["store_id", "week_end_date"])
    qc = {"output_rows": len(out), "stores": out["store_id"].nunique()}
    return out.reset_index(drop=True), qc


def curate_reports(df: pd.DataFrame, valid_store_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = _drop_denormalized(df)
    out = _ensure_columns(out, REPORTS_COLS)
    out = _parse_dates(out, ["report_date"])
    out = out.drop_duplicates(subset=["report_id"])
    out = out[out["store_id"].isin(valid_store_ids)]
    qc = {"output_rows": len(out)}
    return out.reset_index(drop=True), qc


def curate_products(df: pd.DataFrame, valid_store_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = _drop_denormalized(df)
    out = _ensure_columns(out, PRODUCT_COLS)
    out = out.drop_duplicates(subset=["sku_id"])
    out = out[out["store_id"].isin(valid_store_ids)]
    out["in_stock"] = out["in_stock"].map(
        lambda x: x is True or str(x).lower() in ("true", "1", "yes")
    )
    for col in ("unit_price", "margin_pct", "facings", "weekly_units_sold"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    qc = {"output_rows": len(out), "stores": out["store_id"].nunique()}
    return out.reset_index(drop=True), qc


def run_curation(
    scraped_path: Path,
    synthetic_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Full Stage 4 pipeline — read Excel, curate, write CSV + QC manifest."""
    scraped = load_workbook_sheets(scraped_path)
    synthetic = load_workbook_sheets(synthetic_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # Metadata / audit sheets (not fed to StoreDNA model)
    if "scrape_run_log" in scraped:
        scraped["scrape_run_log"].to_csv(meta_dir / "scrape_run_log.csv", index=False)
    if "data_dictionary" in synthetic:
        synthetic["data_dictionary"].to_csv(meta_dir / "data_dictionary.csv", index=False)

    dim_store, dim_qc = build_dim_store(
        scraped["store_catalog"], synthetic["store_catalog"]
    )
    valid_ids = set(dim_store["store_id"])

    stg_reviews, reviews_qc = curate_reviews(scraped["customer_reviews"], valid_ids)
    stg_news, news_qc = curate_news(scraped["local_news"], valid_ids)
    fact_ops, ops_qc = curate_operations_weekly(synthetic["operations_weekly"], valid_ids)
    stg_reports, reports_qc = curate_reports(synthetic["operational_reports"], valid_ids)
    stg_products, products_qc = curate_products(synthetic["product_descriptions"], valid_ids)

    dim_store.to_csv(output_dir / "dim_store.csv", index=False)
    stg_reviews.to_csv(output_dir / "stg_reviews.csv", index=False)
    stg_news.to_csv(output_dir / "stg_news.csv", index=False)
    fact_ops.to_csv(output_dir / "fact_operations_weekly.csv", index=False)
    stg_reports.to_csv(output_dir / "stg_reports.csv", index=False)
    stg_products.to_csv(output_dir / "stg_products.csv", index=False)

    manifest = {
        "curated_at": date.today().isoformat(),
        "sources": {
            "scraped_workbook": str(scraped_path),
            "synthetic_workbook": str(synthetic_path),
        },
        "outputs": {
            "dim_store": len(dim_store),
            "stg_reviews": len(stg_reviews),
            "stg_news": len(stg_news),
            "fact_operations_weekly": len(fact_ops),
            "stg_reports": len(stg_reports),
            "stg_products": len(stg_products),
        },
        "quality_checks": {
            "dim_store": dim_qc,
            "stg_reviews": reviews_qc,
            "stg_news": news_qc,
            "fact_operations_weekly": ops_qc,
            "stg_reports": reports_qc,
            "stg_products": products_qc,
        },
        "notes": [
            "Meta sheets (scrape_run_log, data_dictionary) saved under curated/meta/",
            "Denormalized store columns removed from synthetic fact/staging tables",
            "review_date reflects scrape run date until source dates are parsed",
        ],
    }
    (output_dir / "curation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
