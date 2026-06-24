"""
Scrape real retailer pages and export CSVs in the same format as data/synthetic/.

Usage:
  python scripts/scrape_to_synthetic.py
  python scripts/scrape_to_synthetic.py --skip-llm   # JSON-LD only, no Azure OpenAI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.store_scraper import extract_full_synthetic_profile, scrape
from src.synthetic_mapper import (
    assortment_from_llm,
    demographics_from_llm,
    demographics_from_scrape_text,
    empty_operations_weekly,
    empty_sales_weekly,
    images_from_llm,
    news_from_llm,
    news_from_rss_scrape,
    products_from_llm,
    reports_from_llm,
    reviews_from_llm,
    reviews_from_scrape_text,
    store_master_from_scrape,
    write_synthetic_tables,
    _services_from_text,
)


def load_cached_retailer_scrape(store_id: str) -> dict | None:
    """Use prior successful scrape when live retailer page is thin or blocked."""
    cache_path = PROJECT_ROOT / "data" / "external" / "top_retailers" / f"{store_id}.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def merge_retailer_scrape(live: dict, cached: dict | None) -> dict:
    if not cached:
        return live
    if live.get("scrape_quality") in ("error", "blocked"):
        return {**cached, "_from_cache": True}
    if live.get("char_count", 0) < 200 and cached.get("char_count", 0) > live.get("char_count", 0):
        merged = {**live, **cached}
        merged["_from_cache"] = True
        return merged
    return live


def load_url_inventory(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"store_id", "retailer", "source_type", "url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    return df


def scrape_store_sources(df: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """Return store_id -> source_type -> scrape dict."""
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for _, row in df.iterrows():
        store_id = str(row["store_id"])
        source_type = str(row["source_type"])
        url = str(row["url"]).strip()
        print(f"  scraping {store_id} | {source_type} | {url}")
        try:
            grouped[store_id][source_type] = scrape(url, timeout=35)
            grouped[store_id][source_type]["_retailer"] = str(row["retailer"])
            grouped[store_id][source_type]["_city"] = str(row.get("city", ""))
            grouped[store_id][source_type]["_state"] = str(row.get("state", ""))
        except Exception as exc:
            print(f"    failed: {exc}")
            grouped[store_id][source_type] = {
                "error": str(exc),
                "url": url,
                "text": "",
                "json_ld": [],
                "scrape_quality": "error",
            }
    return grouped


def profile_without_llm(store_id: str, retailer: str, retailer_scrape: dict) -> dict:
    """Minimal profile when LLM is unavailable — JSON-LD + page text heuristics."""
    text = retailer_scrape.get("text", "")
    departments = []
    for block in retailer_scrape.get("json_ld", []):
        for dept in block.get("department", []) or []:
            if isinstance(dept, dict) and dept.get("name"):
                departments.append(dept["name"])

    if not departments:
        departments = _services_from_text(text)

    assortment = []
    for name in departments[:8]:
        assortment.append({
            "category": name,
            "sku_count": "",
            "avg_facings": "",
            "private_label_share": "",
        })

    products = []
    for name in departments[:12]:
        products.append({
            "category": name,
            "product_name": name,
            "description": f"Department/service listed on {retailer} store page.",
            "attributes": [],
            "brand_type": "national",
            "in_stock": True,
            "price_tier": "mid",
        })

    return {
        "store_master": {},
        "demographics": {},
        "customer_reviews": [],
        "assortment_snapshot": assortment,
        "product_descriptions": products,
        "local_news": [],
        "operational_reports": [],
        "image_audits": [],
        "_note": "LLM skipped — partial extraction from JSON-LD and department list",
        "_text_preview": text[:300],
    }


def build_tables_for_store(
    store_id: str,
    retailer: str,
    city: str,
    state: str,
    retailer_scrape: dict,
    demographics_scrape: dict | None,
    reviews_scrape: dict | None,
    news_scrape: dict | None,
    profile: dict,
    sku_start: int,
    news_start: int,
) -> dict[str, list[dict]]:
    store_row = store_master_from_scrape(
        store_id, retailer, retailer_scrape, profile.get("store_master")
    )
    if store_row.get("city"):
        city = store_row["city"]
    if store_row.get("state"):
        state = store_row["state"]
    elif city and not store_row.get("city"):
        store_row["city"] = city
    if state and not store_row.get("state"):
        store_row["state"] = state

    demo_row = demographics_from_llm(store_id, profile.get("demographics", {}))
    if demographics_scrape and (
        demographics_scrape.get("text") or demographics_scrape.get("structured_stats")
    ):
        parsed_demo = demographics_from_scrape_text(
            store_id,
            demographics_scrape.get("text", ""),
            demographics_scrape.get("structured_stats"),
        )
        for key, val in parsed_demo.items():
            if key.startswith("_"):
                continue
            if val and (not demo_row.get(key) or demo_row.get(key) == "unknown"):
                demo_row[key] = val

    review_rows = reviews_from_llm(store_id, profile.get("customer_reviews", []))
    if reviews_scrape and reviews_scrape.get("text"):
        scraped_reviews = reviews_from_scrape_text(
            store_id,
            reviews_scrape["text"],
            source="sitejabber",
            retailer=retailer,
        )
        if scraped_reviews:
            review_rows = scraped_reviews

    news_rows = news_from_llm(
        store_id, store_row.get("city", city), store_row.get("state", state),
        profile.get("local_news", []), start_idx=news_start,
    )
    if news_scrape and news_scrape.get("rss_items"):
        news_rows = news_from_rss_scrape(
            store_id,
            store_row.get("city", city),
            store_row.get("state", state),
            news_scrape,
            start_idx=news_start,
        )

    return {
        "store_master": [store_row],
        "demographics": [demo_row],
        "customer_reviews": review_rows,
        "assortment_snapshot": assortment_from_llm(store_id, profile.get("assortment_snapshot", [])),
        "product_descriptions": products_from_llm(
            store_id, profile.get("product_descriptions", []), start_idx=sku_start
        ),
        "local_news": news_rows,
        "image_audits": images_from_llm(store_id, profile.get("image_audits", [])),
        "operational_reports": reports_from_llm(store_id, profile.get("operational_reports", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape retailers into synthetic CSV format")
    parser.add_argument(
        "--urls",
        type=Path,
        default=PROJECT_ROOT / "data" / "top_retailer_urls.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "scraped",
    )
    parser.add_argument("--skip-llm", action="store_true", help="Use JSON-LD only")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    use_llm = not args.skip_llm
    client = None
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")

    if use_llm:
        key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if key and endpoint:
            client = AzureOpenAI(
                api_key=key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=endpoint,
            )
            print("Azure OpenAI enabled — full synthetic profile extraction")
        else:
            print("No Azure credentials — falling back to JSON-LD only")
            use_llm = False

    print(f"Loading URLs from {args.urls}")
    url_df = load_url_inventory(args.urls)
    scraped = scrape_store_sources(url_df)

    all_tables: dict[str, list[dict]] = defaultdict(list)
    sku_counter = 1
    news_counter = 1
    run_log = []

    for store_id, sources in scraped.items():
        live_retailer = sources.get("retailer_page", {})
        cached = load_cached_retailer_scrape(store_id)
        retailer_scrape = merge_retailer_scrape(live_retailer, cached)
        for key in ("_retailer", "_city", "_state"):
            if live_retailer.get(key):
                retailer_scrape[key] = live_retailer[key]
        demographics_scrape = sources.get("demographics")
        reviews_scrape = sources.get("reviews")
        news_scrape = sources.get("local_news")
        retailer = retailer_scrape.get("_retailer") or sources.get("demographics", {}).get("_retailer", "")
        city = retailer_scrape.get("_city", "")
        state = retailer_scrape.get("_state", "")

        if retailer_scrape.get("scrape_quality") == "error":
            run_log.append({"store_id": store_id, "status": "retailer_page_failed"})
            continue

        if use_llm and client:
            try:
                profile = extract_full_synthetic_profile(
                    client, deployment, retailer, store_id,
                    retailer_scrape, demographics_scrape,
                )
            except Exception as exc:
                print(f"  LLM failed for {store_id}: {exc}")
                profile = profile_without_llm(store_id, retailer, retailer_scrape)
        else:
            profile = profile_without_llm(store_id, retailer, retailer_scrape)

        tables = build_tables_for_store(
            store_id, retailer, city, state,
            retailer_scrape, demographics_scrape, reviews_scrape, news_scrape,
            profile,
            sku_start=sku_counter, news_start=news_counter,
        )
        sku_counter += len(tables["product_descriptions"])
        news_counter += len(tables["local_news"])

        for table_name, rows in tables.items():
            all_tables[table_name].extend(rows)

        run_log.append({
            "store_id": store_id,
            "retailer": retailer,
            "retailer_quality": retailer_scrape.get("scrape_quality"),
            "reviews": len(tables["customer_reviews"]),
            "local_news": len(tables["local_news"]),
            "products": len(tables["product_descriptions"]),
            "assortment": len(tables["assortment_snapshot"]),
        })

        # Save raw scrape bundle per store
        raw_dir = args.output / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{store_id}.json").write_text(
            json.dumps({
                "store_id": store_id,
                "profile": profile,
                "retailer_scrape": {k: v for k, v in retailer_scrape.items() if not k.startswith("_")},
                "demographics_scrape": demographics_scrape,
            }, indent=2, default=str),
            encoding="utf-8",
        )

    output_tables = {
        "store_master": pd.DataFrame(all_tables["store_master"]),
        "demographics": pd.DataFrame(all_tables["demographics"]),
        "customer_reviews": pd.DataFrame(all_tables["customer_reviews"]),
        "assortment_snapshot": pd.DataFrame(all_tables["assortment_snapshot"]),
        "product_descriptions": pd.DataFrame(all_tables["product_descriptions"]),
        "local_news": pd.DataFrame(all_tables["local_news"]),
        "image_audits": pd.DataFrame(all_tables["image_audits"]),
        "operational_reports": pd.DataFrame(all_tables["operational_reports"]),
        "sales_weekly": empty_sales_weekly(),
        "operations_weekly": empty_operations_weekly(),
    }

    write_synthetic_tables(args.output, output_tables)

    metadata = {
        "generated_at": date.today().isoformat(),
        "source": "web_scrape",
        "url_inventory": str(args.urls),
        "stores_processed": run_log,
        "notes": [
            "sales_weekly and operations_weekly require internal POS/ops feeds — empty in scraped output",
            "customer_reviews sourced from public review-aggregator pages (brand-level when store pages block bots)",
            "demographics population_5mi uses city-level Data USA stats as trade-area proxy",
            "local_news from Google News RSS for store city",
        ],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(run_log).to_csv(args.output / "scrape_run_log.csv", index=False)

    print(f"\nWrote synthetic-format CSVs to {args.output}")
    for name, df in output_tables.items():
        print(f"  {name}.csv  {len(df):>4} rows")


if __name__ == "__main__":
    main()
