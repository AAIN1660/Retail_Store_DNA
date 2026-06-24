"""
Collect customer reviews and local news for US retail stores from multiple public sources.

This collector intentionally avoids Google API-key dependencies for store-level scraping.
It uses public review aggregators and RSS feeds only.

Sources:
  Reviews: Sitejabber, Trustpilot, ResellerRatings-style public brand pages
  News: Google News RSS, Bing News RSS (city + retailer + retail keywords)
"""

from __future__ import annotations

import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import pandas as pd
import requests

from src.store_scraper import scrape
from src.synthetic_mapper import news_from_rss_scrape, reviews_from_scrape_text

# Known US retail brands → public review aggregator slugs
RETAILER_REVIEW_SOURCES: dict[str, list[dict[str, str]]] = {
    "Walmart": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/walmart.com"},
        {"source": "trustpilot", "url": "https://www.trustpilot.com/review/www.walmart.com"},
    ],
    "Target": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/target.com"},
        {"source": "trustpilot", "url": "https://www.trustpilot.com/review/www.target.com"},
    ],
    "Kroger": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/kroger.com"},
        {"source": "trustpilot", "url": "https://www.trustpilot.com/review/www.kroger.com"},
    ],
    "Costco": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/costco.com"},
        {"source": "trustpilot", "url": "https://www.trustpilot.com/review/www.costco.com"},
    ],
    "Albertsons": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/albertsons.com"},
    ],
    "Publix": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/publix.com"},
    ],
    "Whole Foods": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/amazon.com"},
        {"source": "trustpilot", "url": "https://www.trustpilot.com/review/www.amazon.com"},
    ],
    "CVS": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/cvs.com"},
    ],
    "Walgreens": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/walgreens.com"},
    ],
    "Home Depot": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/homedepot.com"},
        {"source": "trustpilot", "url": "https://www.trustpilot.com/review/www.homedepot.com"},
    ],
    "Lowe's": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/lowes.com"},
    ],
    "Best Buy": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/bestbuy.com"},
    ],
    "Dollar General": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/dollargeneral.com"},
    ],
    "Sam's Club": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/samsclub.com"},
    ],
    "Aldi": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/aldi.us"},
    ],
    "H-E-B": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/heb.com"},
    ],
    "Meijer": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/meijer.com"},
    ],
    "Safeway": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/safeway.com"},
    ],
    "Sprouts": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/sprouts.com"},
    ],
    "Trader Joe's": [
        {"source": "sitejabber", "url": "https://www.sitejabber.com/reviews/traderjoes.com"},
    ],
}

US_CITIES: list[tuple[str, str]] = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"), ("Houston", "TX"),
    ("Phoenix", "AZ"), ("Philadelphia", "PA"), ("San Antonio", "TX"), ("San Diego", "CA"),
    ("Dallas", "TX"), ("Austin", "TX"), ("Jacksonville", "FL"), ("Fort Worth", "TX"),
    ("Columbus", "OH"), ("Charlotte", "NC"), ("Indianapolis", "IN"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Boston", "MA"), ("Nashville", "TN"), ("Detroit", "MI"),
    ("Portland", "OR"), ("Memphis", "TN"), ("Louisville", "KY"), ("Baltimore", "MD"),
    ("Milwaukee", "WI"), ("Albuquerque", "NM"), ("Tucson", "AZ"), ("Fresno", "CA"),
    ("Sacramento", "CA"), ("Kansas City", "MO"), ("Atlanta", "GA"), ("Miami", "FL"),
    ("Tampa", "FL"), ("Orlando", "FL"), ("Cleveland", "OH"), ("Pittsburgh", "PA"),
    ("St. Louis", "MO"), ("Cincinnati", "OH"), ("Raleigh", "NC"), ("New Orleans", "LA"),
    ("Salt Lake City", "UT"), ("Birmingham", "AL"), ("Richmond", "VA"), ("Boise", "ID"),
    ("Omaha", "NE"), ("Oklahoma City", "OK"), ("Minneapolis", "MN"), ("Buffalo", "NY"),
    ("Rochester", "NY"), ("Spokane", "WA"), ("Tulsa", "OK"), ("Honolulu", "HI"),
]

RETAILERS_ROTATION = list(RETAILER_REVIEW_SOURCES.keys())


@dataclass
class CollectorConfig:
    """Runtime settings for review/news collection."""

    sleep_seconds: float = 1.2
    max_reviews_per_source: int = 8
    max_news_per_feed: int = 6
    request_timeout: int = 30


@dataclass
class ScrapeLogRow:
    store_id: str
    retailer: str
    city: str
    state: str
    source_type: str
    source_name: str
    url: str
    status: str
    records: int = 0
    message: str = ""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_us_retail_store_catalog(n_stores: int = 100) -> pd.DataFrame:
    """Build n_stores across known US retail brands and major cities."""
    rows: list[dict[str, str]] = []
    retailers = RETAILERS_ROTATION
    for i in range(n_stores):
        retailer = retailers[i % len(retailers)]
        city, state = US_CITIES[i % len(US_CITIES)]
        store_id = f"USR-{i + 1:03d}"
        rows.append({
            "store_id": store_id,
            "retailer": retailer,
            "banner": retailer,
            "store_name": f"{retailer} {city}",
            "city": city,
            "state": state,
            "store_format": "Supercenter" if retailer in ("Walmart", "Costco", "Sam's Club") else "Neighborhood",
        })
    return pd.DataFrame(rows)


def google_news_rss_url(query: str) -> str:
    q = urllib.parse.quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def bing_news_rss_url(query: str) -> str:
    q = urllib.parse.quote_plus(query)
    return f"https://www.bing.com/news/search?q={q}&format=rss"


def news_feed_urls(store: dict[str, str]) -> list[dict[str, str]]:
    """RSS feeds for local retail news around a store."""
    city = store["city"]
    state = store["state"]
    retailer = store["retailer"]
    loc = f"{city} {state}"
    return [
        {
            "source_name": "google_news_retail",
            "url": google_news_rss_url(f"{loc} {retailer} retail store"),
        },
        {
            "source_name": "google_news_grocery",
            "url": google_news_rss_url(f"{loc} grocery shopping retail"),
        },
        {
            "source_name": "google_news_local",
            "url": google_news_rss_url(f"{loc} retail opening OR store closing"),
        },
        {
            "source_name": "bing_news_retail",
            "url": bing_news_rss_url(f"{loc} {retailer} retail"),
        },
    ]


def review_source_urls(store: dict[str, str]) -> list[dict[str, str]]:
    """Brand-level review pages for the store's retailer."""
    retailer = store["retailer"]
    sources = list(RETAILER_REVIEW_SOURCES.get(retailer, []))
    city = store["city"]
    state = store["state"]
    # City-specific Google News sometimes surfaces local review roundups
    sources.append({
        "source": "google_news_reviews",
        "url": google_news_rss_url(f"{city} {state} {retailer} customer reviews"),
    })
    return sources


def _safe_scrape(url: str, timeout: int) -> dict[str, Any]:
    try:
        return scrape(url, timeout=timeout)
    except Exception as exc:
        return {
            "url": url,
            "text": "",
            "rss_items": [],
            "scrape_quality": "error",
            "error": str(exc),
        }


def collect_reviews_for_store(
    store: dict[str, str],
    config: CollectorConfig,
) -> tuple[list[dict[str, Any]], list[ScrapeLogRow]]:
    """Scrape review-like content from all configured sources for one store."""
    rows: list[dict[str, Any]] = []
    logs: list[ScrapeLogRow] = []
    seen_text: set[str] = set()

    for src in review_source_urls(store):
        url = src["url"]
        source_name = src["source"]
        print(f"    reviews | {source_name} | {url[:70]}...")

        if source_name == "google_news_reviews":
            article = _safe_scrape(url, config.request_timeout)
            # Treat RSS review roundups as pseudo-reviews from headlines/summaries
            for item in article.get("rss_items") or []:
                text = f"{item.get('headline', '')}. {item.get('summary', '')}".strip()
                if len(text) < 40:
                    continue
                key = text[:80].lower()
                if key in seen_text:
                    continue
                seen_text.add(key)
                extracted = reviews_from_scrape_text(
                    store["store_id"], text, source=source_name, retailer=store["retailer"]
                )
                if not extracted:
                    extracted = [{
                        "review_id": f"REV-{store['store_id']}-{source_name[:2]}-{len(rows)+1:03d}",
                        "store_id": store["store_id"],
                        "review_date": date.today().isoformat(),
                        "source": source_name,
                        "rating": "",
                        "sentiment": "neutral",
                        "review_text": text[:500],
                        "complaint_tags": "",
                    }]
                rows.extend(extracted[:3])
            status = "ok" if article.get("rss_items") else article.get("scrape_quality", "empty")
            logs.append(ScrapeLogRow(
                store_id=store["store_id"], retailer=store["retailer"],
                city=store["city"], state=store["state"],
                source_type="reviews", source_name=source_name, url=url,
                status=status, records=len(article.get("rss_items") or []),
            ))
        else:
            article = _safe_scrape(url, config.request_timeout)
            extracted = reviews_from_scrape_text(
                store["store_id"],
                article.get("text", ""),
                source=source_name,
                retailer=store["retailer"],
            )
            added = 0
            for row in extracted[: config.max_reviews_per_source]:
                key = row["review_text"][:80].lower()
                if key in seen_text:
                    continue
                seen_text.add(key)
                row["source"] = source_name
                rows.append(row)
                added += 1
            logs.append(ScrapeLogRow(
                store_id=store["store_id"], retailer=store["retailer"],
                city=store["city"], state=store["state"],
                source_type="reviews", source_name=source_name, url=url,
                status=article.get("scrape_quality", "ok"), records=added,
                message=article.get("error", ""),
            ))

        time.sleep(config.sleep_seconds)

    return rows, logs


def collect_news_for_store(
    store: dict[str, str],
    config: CollectorConfig,
    news_id_start: int = 1,
) -> tuple[list[dict[str, Any]], list[ScrapeLogRow], int]:
    """Scrape local news RSS feeds for one store."""
    rows: list[dict[str, Any]] = []
    logs: list[ScrapeLogRow] = []
    seen_headlines: set[str] = set()
    next_id = news_id_start

    for feed in news_feed_urls(store):
        url = feed["url"]
        source_name = feed["source_name"]
        print(f"    news | {source_name} | {url[:70]}...")

        article = _safe_scrape(url, config.request_timeout)
        batch = news_from_rss_scrape(
            store["store_id"],
            store["city"],
            store["state"],
            article,
            start_idx=next_id,
        )
        added = 0
        for row in batch[: config.max_news_per_feed]:
            key = row["headline"].lower().strip()
            if key in seen_headlines:
                continue
            seen_headlines.add(key)
            row["news_id"] = f"NEWS-{next_id:05d}"
            next_id += 1
            rows.append(row)
            added += 1

        logs.append(ScrapeLogRow(
            store_id=store["store_id"], retailer=store["retailer"],
            city=store["city"], state=store["state"],
            source_type="local_news", source_name=source_name, url=url,
            status=article.get("scrape_quality", "ok"), records=added,
            message=article.get("error", ""),
        ))
        time.sleep(config.sleep_seconds)

    return rows, logs, next_id


def build_url_inventory(stores: pd.DataFrame) -> pd.DataFrame:
    """Flat URL list for all stores — reviews + news sources."""
    inv_rows: list[dict[str, str]] = []
    for _, store in stores.iterrows():
        s = store.to_dict()
        for src in review_source_urls(s):
            inv_rows.append({
                "store_id": s["store_id"],
                "retailer": s["retailer"],
                "city": s["city"],
                "state": s["state"],
                "source_type": "reviews",
                "source_name": src["source"],
                "url": src["url"],
            })
        for feed in news_feed_urls(s):
            inv_rows.append({
                "store_id": s["store_id"],
                "retailer": s["retailer"],
                "city": s["city"],
                "state": s["state"],
                "source_type": "local_news",
                "source_name": feed["source_name"],
                "url": feed["url"],
            })
    return pd.DataFrame(inv_rows)


def collect_all_stores(
    stores: pd.DataFrame,
    config: CollectorConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Collect reviews and news for all stores.

    Returns (customer_reviews_df, local_news_df, scrape_log_df).
    """
    all_reviews: list[dict[str, Any]] = []
    all_news: list[dict[str, Any]] = []
    all_logs: list[ScrapeLogRow] = []
    news_id_counter = 1
    total = len(stores)

    for idx, (_, store_row) in enumerate(stores.iterrows(), start=1):
        store = store_row.to_dict()
        store_id = store["store_id"]
        if progress_callback:
            progress_callback(idx, total, store_id)
        else:
            print(f"\n[{idx}/{total}] {store_id} | {store['retailer']} | {store['city']}, {store['state']}")

        rev_rows, rev_logs = collect_reviews_for_store(store, config)
        all_reviews.extend(rev_rows)
        all_logs.extend(rev_logs)

        news_rows, news_logs, news_id_counter = collect_news_for_store(
            store, config, news_id_start=news_id_counter
        )
        all_news.extend(news_rows)
        all_logs.extend(news_logs)

    reviews_df = pd.DataFrame(all_reviews)
    news_df = pd.DataFrame(all_news)
    log_df = pd.DataFrame([row.__dict__ for row in all_logs])
    return reviews_df, news_df, log_df
