"""
Map scraped web content into the same table shapes as data/synthetic/*.csv
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

# Column order matches synthetic generator output
STORE_MASTER_COLS = [
    "store_id", "retailer", "banner", "store_name", "store_format",
    "address", "city", "state", "zip_code", "sq_ft", "open_date",
    "has_pharmacy", "has_online_pickup",
]

DEMOGRAPHICS_COLS = [
    "store_id", "population_5mi", "median_income", "median_age",
    "household_size", "urbanicity", "competitor_count_3mi",
]

REVIEWS_COLS = [
    "review_id", "store_id", "review_date", "source", "rating",
    "sentiment", "review_text", "complaint_tags",
]

ASSORTMENT_COLS = [
    "store_id", "category", "sku_count", "avg_facings", "private_label_share",
]

PRODUCT_COLS = [
    "sku_id", "store_id", "category", "product_name", "description",
    "attributes", "brand_type", "in_stock", "price_tier",
]

NEWS_COLS = [
    "news_id", "store_id", "city", "state", "published_date", "headline",
    "summary", "event_type", "demand_impact", "source",
]

IMAGE_COLS = [
    "image_id", "store_id", "capture_date", "image_zone", "file_path",
    "store_condition_score", "shelf_utilization_pct", "branding_compliance_score",
    "detected_issues", "annotator",
]

REPORT_COLS = [
    "report_id", "store_id", "report_date", "report_type", "severity",
    "issue_category", "description", "status", "reported_by",
]


def _first_json_ld_type(scrape: dict, type_name: str) -> dict | None:
    for block in scrape.get("json_ld", []):
        if block.get("@type") == type_name:
            return block
        if block.get("@type") in ("Store", "DepartmentStore", "GroceryStore", "Supermarket"):
            return block
    return scrape.get("json_ld", [{}])[0] if scrape.get("json_ld") else None


def _address_parts(scrape: dict) -> dict[str, str]:
    block = _first_json_ld_type(scrape, "DepartmentStore") or {}
    addr = block.get("address", {})
    if isinstance(addr, dict):
        return {
            "street": addr.get("streetAddress", ""),
            "city": addr.get("addressLocality", ""),
            "state": addr.get("addressRegion", ""),
            "zip": addr.get("postalCode", ""),
        }
    return {"street": "", "city": "", "state": "", "zip": ""}


def _services_from_text(text: str) -> list[str]:
    """Pull common retail departments/services mentioned in page copy."""
    keywords = [
        "Pharmacy", "Bakery", "Deli", "Grocery", "Produce", "Floral",
        "Butcher", "Fuel Station", "Garden Center", "Vision Center",
        "Starbucks", "DriveUp", "Delivery", "Liquor", "Seafood",
    ]
    found = []
    lower = text.lower()
    for kw in keywords:
        if kw.lower() in lower:
            found.append(kw)
    return found


def store_master_from_scrape(
    store_id: str,
    retailer: str,
    scrape: dict,
    llm_store: dict | None = None,
) -> dict[str, Any]:
    """Build store_master row from JSON-LD + optional LLM fields."""
    block = _first_json_ld_type(scrape, "DepartmentStore") or {}
    addr = _address_parts(scrape)
    llm = llm_store or {}
    text = scrape.get("text", "")
    meta = scrape.get("meta", {})

    services = llm.get("services") or []
    dept_names = []
    for d in block.get("department", []) or []:
        if isinstance(d, dict) and d.get("name"):
            dept_names.append(d["name"])

    if not dept_names:
        dept_names = _services_from_text(text)

    all_services = list({*(services if isinstance(services, list) else []), *dept_names})
    service_blob = " ".join(all_services).lower()

    name = llm.get("store_name") or block.get("name") or scrape.get("title", "")
    if name in ("Untitled", "") and meta.get("og:title"):
        name = meta["og:title"]
    fmt = llm.get("store_format") or ""
    if not fmt:
        title_lower = name.lower()
        if "supercenter" in title_lower or "super center" in title_lower:
            fmt = "Supercenter"
        elif "warehouse" in title_lower:
            fmt = "Warehouse"
        elif "express" in title_lower:
            fmt = "Express"
        elif "supermarket" in title_lower:
            fmt = "Supermarket"
        else:
            fmt = "Neighborhood"

    street = llm.get("address") or addr["street"]
    city = llm.get("city") or addr["city"]
    state = llm.get("state") or addr["state"]
    zip_code = llm.get("zip_code") or addr["zip"]

    if not street:
        m = re.search(r"located at ([^,]+), ([^,]+), ([A-Z]{2})\b", text, re.I)
        if m:
            street, city, state = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        else:
            m2 = re.search(r"at (\d+[^,]+), ([^,]+), ([A-Z]{2})\b", text)
            if m2:
                street, city, state = m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()
            else:
                m_tgt = re.search(
                    r"(\d+[^\n,]+)\s*\n\s*([^,]+),\s*([A-Z]{2})\s*(\d{5})?",
                    text,
                )
                if m_tgt:
                    street = m_tgt.group(1).strip()
                    city = m_tgt.group(2).strip()
                    state = m_tgt.group(3).strip()
                    if m_tgt.group(4):
                        zip_code = m_tgt.group(4).strip()

    if not city or not state:
        m3 = re.search(r"Target (\w+) Store, (\w+), ([A-Z]{2})\b", name)
        if m3:
            city, state = m3.group(2), m3.group(3)
        elif scrape.get("title"):
            m4 = re.search(r", (\w+), ([A-Z]{2})\b", scrape["title"])
            if m4 and not city:
                city, state = m4.group(1), m4.group(2)

    if not street and meta.get("description"):
        m = re.search(r"located at ([^,]+)", meta["description"], re.I)
        if m:
            street = m.group(1).strip()

    phones = scrape.get("regex_signals", {}).get("phones_found") or []
    phone = llm.get("phone") or (phones[0] if phones else "")

    return {
        "store_id": store_id,
        "retailer": retailer,
        "banner": retailer,
        "store_name": name[:120],
        "store_format": fmt,
        "address": street,
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "sq_ft": llm.get("sq_ft") or "",
        "open_date": llm.get("open_date") or "",
        "has_pharmacy": any(x in service_blob for x in ("pharmacy", "health")),
        "has_online_pickup": any(
            x in text.lower()
            for x in ("pickup", "delivery", "driveup", "drive up", "online")
        ),
    }


def demographics_from_llm(store_id: str, llm_demo: dict) -> dict[str, Any]:
    return {
        "store_id": store_id,
        "population_5mi": llm_demo.get("population", ""),
        "median_income": llm_demo.get("median_household_income", ""),
        "median_age": llm_demo.get("median_age", ""),
        "household_size": llm_demo.get("household_size", ""),
        "urbanicity": llm_demo.get("urbanicity", "unknown"),
        "competitor_count_3mi": llm_demo.get("competitor_count_3mi", ""),
    }


def demographics_from_scrape_text(
    store_id: str,
    text: str,
    structured_stats: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Heuristic parse of Data USA profile text — no LLM required."""
    if not text and not structured_stats:
        return demographics_from_llm(store_id, {})

    stats = structured_stats or {}

    def _find(pattern: str) -> str:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).replace(",", "") if m else ""

    def _stat_value(*labels: str) -> str:
        for label in labels:
            if label in stats:
                return stats[label].replace(",", "").replace("$", "").replace("%", "").strip()
            for key, val in stats.items():
                if label.lower() in key.lower():
                    return val.replace(",", "").replace("$", "").replace("%", "").strip()
        return ""

    population = _stat_value("Population", "2024 Population")
    if not population:
        population = _find(r"population of ([\d,]+)")
    if not population:
        population = _find(r"had a population of ([\d,]+)")

    income = _stat_value("Median Household Income", "2024 Median Household Income")
    if not income:
        income = _find(r"median household income of \$?([\d,]+)")
    if not income:
        income = _find(r"median household income was \$?([\d,]+)")

    age = _stat_value("Median Age", "2024 Median Age")
    if not age:
        age = _find(r"median age of ([\d.]+)")
    if not age:
        age = _find(r"median age was ([\d.]+)")

    home_value = _stat_value("Median Property Value", "2024 Median Property Value")
    if not home_value:
        home_value = _find(r"median property value of \$?([\d,]+)")
    if not home_value:
        home_value = _find(r"median property value in [^,]+, [A-Z]{2} was \$?([\d,]+)")

    household_size = _find(r"was ([\d.]+) cars per household")
    if not household_size:
        household_size = _find(r"average of ([\d.]+) people per household")

    urbanicity = "unknown"
    if population:
        try:
            pop_n = int(float(population))
            if pop_n >= 500_000:
                urbanicity = "urban"
            elif pop_n <= 20_000:
                urbanicity = "rural"
            else:
                urbanicity = "suburban"
        except ValueError:
            pass
    if urbanicity == "unknown":
        if re.search(r"urban area|urban population", text, re.I):
            urbanicity = "urban"
        elif re.search(r"rural", text, re.I):
            urbanicity = "rural"
        elif re.search(r"suburban", text, re.I):
            urbanicity = "suburban"

    return {
        "store_id": store_id,
        "population_5mi": population,
        "median_income": income,
        "median_age": age,
        "household_size": household_size,
        "urbanicity": urbanicity,
        "competitor_count_3mi": "",
        "_housing_median_value": home_value,
    }


_POSITIVE_WORDS = {
    "great", "excellent", "friendly", "clean", "fast", "helpful", "love",
    "pleasant", "fresh", "organized", "satisfied", "appreciate", "good",
}
_NEGATIVE_WORDS = {
    "dirty", "rude", "slow", "wait", "frustrat", "disappoint", "worst",
    "horrible", "out of stock", "missing", "refund", "complaint", "bad",
}


def _infer_sentiment(text: str) -> str:
    lower = text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in lower)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def reviews_from_scrape_text(
    store_id: str,
    text: str,
    source: str = "web_scrape",
    retailer: str = "",
) -> list[dict]:
    """Extract review-like sentences from review-aggregator page text."""
    if not text:
        return []

    retailer_lower = retailer.lower()
    candidates: list[str] = []

    for chunk in re.split(r"(?<=[.!?])\s+", text):
        chunk = chunk.strip()
        if len(chunk) < 45 or len(chunk) > 450:
            continue
        lower = chunk.lower()
        if retailer_lower and retailer_lower not in lower and "store" not in lower:
            if not re.search(r"\b(i|my|we|our)\b", lower):
                continue
        if any(skip in lower for skip in (
            "summary is generated by ai",
            "we monitor reviews",
            "check out our latest communities",
            "smartcustomer",
            "review guidelines",
        )):
            continue
        if re.search(r"\b(i am|i was|my experience|while \w+|customers?)\b", lower):
            candidates.append(chunk)

    rows = []
    for i, review_text in enumerate(candidates[:12], start=1):
        rating_match = re.search(r"(\d(?:\.\d)?)\s*(?:out of 5|stars?)", review_text, re.I)
        rows.append({
            "review_id": f"REV-{store_id}-{i:03d}",
            "store_id": store_id,
            "review_date": date.today().isoformat(),
            "source": source,
            "rating": rating_match.group(1) if rating_match else "",
            "sentiment": _infer_sentiment(review_text),
            "review_text": review_text[:500],
            "complaint_tags": "",
        })
    return rows


def news_from_rss_scrape(
    store_id: str,
    city: str,
    state: str,
    scrape: dict,
    start_idx: int = 1,
) -> list[dict]:
    """Build local_news rows from parsed RSS items on a scrape dict."""
    items = scrape.get("rss_items") or []
    rows = []
    for i, item in enumerate(items, start=start_idx):
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        event_type = "local_news"
        lower = f"{headline} {summary}".lower()
        if any(w in lower for w in ("storm", "weather", "flood", "heat")):
            event_type = "weather"
        elif any(w in lower for w in ("festival", "parade", "marathon", "5k")):
            event_type = "community_event"
        elif any(w in lower for w in ("construction", "road", "closure")):
            event_type = "construction"
        elif any(w in lower for w in ("opening", "new store", "competitor")):
            event_type = "new_competitor"

        impact = "neutral"
        if event_type in ("weather", "construction", "road_closure"):
            impact = "negative"
        elif event_type in ("community_event", "local_festival"):
            impact = "positive"

        pub = item.get("published_date", "")
        if pub and "," in pub:
            try:
                from email.utils import parsedate_to_datetime
                pub = parsedate_to_datetime(pub).date().isoformat()
            except (ValueError, TypeError, OverflowError):
                pub = date.today().isoformat()
        elif not pub:
            pub = date.today().isoformat()

        rows.append({
            "news_id": f"NEWS-{i:05d}",
            "store_id": store_id,
            "city": city,
            "state": state,
            "published_date": pub,
            "headline": headline,
            "summary": summary or f"Local coverage in {city}, {state}.",
            "event_type": event_type,
            "demand_impact": impact,
            "source": item.get("source", "rss"),
        })
    return rows


def reviews_from_llm(store_id: str, reviews: list[dict]) -> list[dict]:
    rows = []
    for i, rev in enumerate(reviews or [], start=1):
        tags = rev.get("complaint_tags") or []
        if isinstance(tags, list):
            tags = "|".join(tags)
        rows.append({
            "review_id": f"REV-{store_id}-{i:03d}",
            "store_id": store_id,
            "review_date": rev.get("review_date") or date.today().isoformat(),
            "source": rev.get("source") or "web_scrape",
            "rating": rev.get("rating", ""),
            "sentiment": rev.get("sentiment", "unknown"),
            "review_text": rev.get("review_text", ""),
            "complaint_tags": tags,
        })
    return rows


def assortment_from_llm(store_id: str, rows: list[dict]) -> list[dict]:
    out = []
    for row in rows or []:
        out.append({
            "store_id": store_id,
            "category": row.get("category", ""),
            "sku_count": row.get("sku_count", ""),
            "avg_facings": row.get("avg_facings", ""),
            "private_label_share": row.get("private_label_share", ""),
        })
    return out


def products_from_llm(store_id: str, rows: list[dict], start_idx: int = 1) -> list[dict]:
    out = []
    for i, row in enumerate(rows or [], start=start_idx):
        attrs = row.get("attributes") or []
        if isinstance(attrs, list):
            attrs = ",".join(attrs)
        out.append({
            "sku_id": f"SKU-{i:06d}",
            "store_id": store_id,
            "category": row.get("category", ""),
            "product_name": row.get("product_name", ""),
            "description": row.get("description", ""),
            "attributes": attrs,
            "brand_type": row.get("brand_type", ""),
            "in_stock": row.get("in_stock", ""),
            "price_tier": row.get("price_tier", ""),
        })
    return out


def news_from_llm(store_id: str, city: str, state: str, rows: list[dict], start_idx: int = 1) -> list[dict]:
    out = []
    for i, row in enumerate(rows or [], start=start_idx):
        out.append({
            "news_id": f"NEWS-{i:05d}",
            "store_id": store_id,
            "city": row.get("city") or city,
            "state": row.get("state") or state,
            "published_date": row.get("published_date") or date.today().isoformat(),
            "headline": row.get("headline", ""),
            "summary": row.get("summary", ""),
            "event_type": row.get("event_type", ""),
            "demand_impact": row.get("demand_impact", ""),
            "source": row.get("source", "web_scrape"),
        })
    return out


def images_from_llm(store_id: str, rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows or [], start=1):
        issues = row.get("detected_issues") or []
        if isinstance(issues, list):
            issues = "|".join(issues)
        out.append({
            "image_id": f"IMG-{store_id}-{i:03d}",
            "store_id": store_id,
            "capture_date": row.get("capture_date") or date.today().isoformat(),
            "image_zone": row.get("image_zone", ""),
            "file_path": row.get("file_path", ""),
            "store_condition_score": row.get("store_condition_score", ""),
            "shelf_utilization_pct": row.get("shelf_utilization_pct", ""),
            "branding_compliance_score": row.get("branding_compliance_score", ""),
            "detected_issues": issues,
            "annotator": row.get("annotator", "web_scrape"),
        })
    return out


def reports_from_llm(store_id: str, rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows or [], start=1):
        out.append({
            "report_id": f"RPT-{store_id}-{i:03d}",
            "store_id": store_id,
            "report_date": row.get("report_date") or date.today().isoformat(),
            "report_type": row.get("report_type", ""),
            "severity": row.get("severity", ""),
            "issue_category": row.get("issue_category", ""),
            "description": row.get("description", ""),
            "status": row.get("status", ""),
            "reported_by": row.get("reported_by", "web_scrape"),
        })
    return out


def empty_sales_weekly() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "store_id", "week_end_date", "category", "revenue", "units",
        "transactions", "promo_sales_pct",
    ])


def empty_operations_weekly() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "store_id", "week_end_date", "labor_hours", "shrink_pct",
        "oos_rate", "fulfillment_rate", "customer_complaints",
    ])


def write_synthetic_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)


FULL_PROFILE_SCHEMA = """
{
  "store_master": {
    "store_name": "",
    "store_format": "",
    "address": "",
    "city": "",
    "state": "",
    "zip_code": "",
    "sq_ft": "",
    "open_date": "",
    "services": []
  },
  "demographics": {
    "population": "",
    "median_household_income": "",
    "median_age": "",
    "household_size": "",
    "urbanicity": "urban|suburban|rural|unknown",
    "competitor_count_3mi": ""
  },
  "customer_reviews": [
    {
      "review_date": "",
      "source": "",
      "rating": "",
      "sentiment": "positive|neutral|negative",
      "review_text": "",
      "complaint_tags": []
    }
  ],
  "assortment_snapshot": [
    {"category": "", "sku_count": "", "avg_facings": "", "private_label_share": ""}
  ],
  "product_descriptions": [
    {
      "category": "",
      "product_name": "",
      "description": "",
      "attributes": [],
      "brand_type": "",
      "in_stock": true,
      "price_tier": ""
    }
  ],
  "local_news": [
    {
      "headline": "",
      "summary": "",
      "event_type": "",
      "demand_impact": "positive|negative|neutral",
      "source": "",
      "published_date": ""
    }
  ],
  "operational_reports": [
    {
      "report_date": "",
      "report_type": "",
      "severity": "",
      "issue_category": "",
      "description": "",
      "status": ""
    }
  ],
  "image_audits": [
    {
      "image_zone": "",
      "file_path": "",
      "store_condition_score": "",
      "shelf_utilization_pct": "",
      "branding_compliance_score": "",
      "detected_issues": [],
      "capture_date": ""
    }
  ]
}
"""
