"""
StoreDNA web scraper utilities — enhanced beyond basic HTML text extraction.

Handles retailer pages, demographics portals (Data USA), and local context pages.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import unescape
from typing import Any

import requests

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from readability import Document
except ImportError:
    Document = None

# Browser-like headers reduce bot-wall rate on some retail sites
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_TEXT_CHARS = 12_000
BOT_WALL_PHRASES = ("robot or human", "access denied", "captcha", "unsupported browser")


def validate_url(url: str) -> None:
    if not re.match(r"^https?://", url.strip()):
        raise ValueError(f"Invalid URL: {url}")


def _clean_html_text(raw: str) -> str:
    text = unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", text).strip()


def _extract_json_ld(html: str) -> list[dict[str, Any]]:
    """Pull schema.org JSON-LD blocks embedded in the page."""
    blocks: list[dict[str, Any]] = []
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE,
    ):
        try:
            payload = json.loads(match.group(1).strip())
            if isinstance(payload, list):
                blocks.extend(payload)
            else:
                blocks.append(payload)
        except json.JSONDecodeError:
            continue
    return blocks


def _extract_meta(html: str) -> dict[str, str]:
    """Read common meta / OpenGraph tags."""
    meta: dict[str, str] = {}
    for tag in re.findall(r"<meta[^>]+>", html, re.IGNORECASE):
        name = re.search(r'name=["\']([^"\']+)["\']', tag, re.I)
        prop = re.search(r'property=["\']([^"\']+)["\']', tag, re.I)
        content = re.search(r'content=["\']([^"\']*)["\']', tag, re.I)
        if content:
            key = (name or prop)
            if key:
                meta[key.group(1).lower()] = content.group(1)
    return meta


def _regex_signals(text: str) -> dict[str, Any]:
    """Heuristic extraction of address / phone / hours from visible text."""
    phones = re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    hours = re.findall(
        r"(?:Opens?|Open|Hours?)[:\s]+[^|\n]{5,40}",
        text,
        re.IGNORECASE,
    )
    return {
        "phones_found": phones[:3],
        "hours_snippets": hours[:3],
    }


def _is_bot_wall(title: str, text: str) -> bool:
    blob = f"{title} {text}".lower()
    return any(phrase in blob for phrase in BOT_WALL_PHRASES)


def _parse_datausa_stats(html: str) -> dict[str, str]:
    """Pull headline stats from Data USA profile HTML."""
    stats: dict[str, str] = {}
    for match in re.finditer(
        r'<div class="stat-title">([^<]+)</div>\s*<div class="stat-value">([^<]+)</div>',
        html,
        re.IGNORECASE,
    ):
        stats[match.group(1).strip()] = match.group(2).strip()
    return stats


def _parse_rss_items(xml_text: str, limit: int = 8) -> list[dict[str, str]]:
    """Extract news headlines from RSS/XML feeds (Google News, Bing News, etc.)."""
    import xml.etree.ElementTree as ET

    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    for item in root.findall(".//item")[:limit]:
        row: dict[str, str] = {}
        for child in item:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            row[tag] = (child.text or "").strip()

        title = _clean_html_text(row.get("title", ""))
        if not title:
            continue
        summary = _clean_html_text(row.get("description", ""))
        published = row.get("pubDate", "")
        link = row.get("link", "")
        source = row.get("source", "")
        if not source and link:
            source = re.sub(r"^https?://", "", link).split("/")[0].replace("www.", "")
        items.append({
            "headline": title[:200],
            "summary": summary[:400],
            "published_date": published,
            "source": source or "rss",
        })
    return items


def scrape(url: str, timeout: int = 25) -> dict[str, Any]:
    """
    Fetch a public URL and return cleaned text plus structured hints.

    Returns dict with: url, title, text, json_ld, meta, regex_signals,
    scrape_quality, scraped_at, char_count
    """
    validate_url(url)

    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    html = response.text

    text = None
    if trafilatura:
        text = trafilatura.extract(html, include_comments=False, include_tables=True)

    if not text and Document:
        text = _clean_html_text(Document(html).summary())

    if not text:
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL | re.IGNORECASE)
        text = _clean_html_text(" ".join(paragraphs))

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    title = _clean_html_text(title_match.group(1)) if title_match else "Untitled"

    json_ld = _extract_json_ld(html)
    meta = _extract_meta(html)
    regex_signals = _regex_signals(text or "")
    structured_stats: dict[str, str] = {}
    rss_items: list[dict[str, str]] = []

    if "datausa.io" in url:
        structured_stats = _parse_datausa_stats(html)
        if structured_stats:
            stat_lines = [f"{k}: {v}" for k, v in structured_stats.items()]
            text = "\n".join(stat_lines) + "\n\n" + (text or "")

    if url.endswith(".rss") or "/rss" in url or "format=rss" in url:
        rss_items = _parse_rss_items(html)
        if rss_items:
            rss_lines = [
                f"- {row['headline']} ({row.get('published_date', '')})"
                for row in rss_items
            ]
            text = "Local news headlines:\n" + "\n".join(rss_lines) + "\n\n" + (text or "")

    # Enrich text with JSON-LD summary for LLM when page body is thin
    if json_ld:
        ld_text = json.dumps(json_ld, indent=2)[:4000]
        text = f"{text or ''}\n\n--- structured data ---\n{ld_text}".strip()

    text = (text or "")[:MAX_TEXT_CHARS]
    bot_wall = _is_bot_wall(title, text)

    if bot_wall:
        quality = "blocked"
    elif len(text) < 120:
        quality = "low"
    elif len(text) < 400:
        quality = "medium"
    else:
        quality = "high"

    return {
        "url": url,
        "title": title,
        "text": text,
        "json_ld": json_ld,
        "meta": meta,
        "regex_signals": regex_signals,
        "structured_stats": structured_stats,
        "rss_items": rss_items,
        "scrape_quality": quality,
        "char_count": len(text),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def _strip_json_fences(content: str) -> str:
    content = re.sub(r"^```json\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def build_prompt(source_type: str, article: dict, store_id: str, retailer: str) -> str:
    """Return the LLM prompt for a given source type."""
    common_ctx = f"""
Context:
- store_id: {store_id or "unknown"}
- retailer: {retailer or "unknown"}
- source_type: {source_type}
- page_url: {article["url"]}
- page_title: {article["title"]}
- scrape_quality: {article.get("scrape_quality", "unknown")}
"""

    if source_type == "demographics":
        schema = """
{
  "city": "",
  "state": "",
  "population": "",
  "median_household_income": "",
  "median_age": "",
  "poverty_rate": "",
  "employment_rate": "",
  "education_college_pct": "",
  "housing_median_value": "",
  "urbanicity": "urban|suburban|rural|unknown",
  "summary": "",
  "confidence": "high|medium|low"
}
"""
        task = "Extract trade-area demographics for Store DNA. Use only numbers present in the page."

    elif source_type == "reviews":
        schema = """
{
  "store_name": "",
  "avg_rating": "",
  "review_count": "",
  "sentiment": "positive|neutral|negative|unknown",
  "review_themes": [],
  "complaint_tags": [],
  "positive_themes": [],
  "summary": "",
  "confidence": "high|medium|low"
}
"""
        task = "Extract customer review signals. If no reviews are on the page, return unknown/low confidence."

    elif source_type == "local_context":
        schema = """
{
  "city": "",
  "state": "",
  "local_keywords": [],
  "notable_features": [],
  "economic_drivers": [],
  "event_risk_factors": [],
  "summary": "",
  "confidence": "high|medium|low"
}
"""
        task = "Extract local market context useful for retail demand forecasting."

    else:  # retailer_page (default)
        schema = """
{
  "store_name": "",
  "retailer": "",
  "address": "",
  "city": "",
  "state": "",
  "zip_code": "",
  "phone": "",
  "store_format": "",
  "services": [],
  "hours": "",
  "promotions_mentioned": [],
  "competitive_signals": [],
  "summary": "",
  "confidence": "high|medium|low"
}
"""
        task = "Extract store identity, location, hours, and services from a retailer store page."

    return f"""
You are building data for a Retail Store DNA product.
{task}

{common_ctx}

Rules:
- Return ONLY valid JSON matching the schema below.
- Do not invent facts; use empty strings/lists if missing.
- Lower confidence when scrape_quality is low or blocked.

JSON schema:
{schema}

Regex hints: {json.dumps(article.get("regex_signals", {}))}

Page content:
{article["text"]}
"""


def analyze_scraped_page(
    client: Any,
    deployment: str,
    article: dict,
    store_id: str = "",
    retailer: str = "",
    source_type: str = "retailer_page",
) -> dict:
    """Use Azure OpenAI to structure scraped page content."""
    if not article.get("text"):
        return {
            "error": "No text extracted from page",
            "confidence": "low",
            "_meta": {
                "store_id": store_id,
                "retailer": retailer,
                "source_type": source_type,
                "url": article.get("url"),
                "scrape_quality": article.get("scrape_quality"),
            },
        }

    prompt = build_prompt(source_type, article, store_id, retailer)

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    content = _strip_json_fences(response.choices[0].message.content)
    parsed = json.loads(content)
    parsed["_meta"] = {
        "store_id": store_id,
        "retailer": retailer,
        "source_type": source_type,
        "url": article["url"],
        "page_title": article["title"],
        "scrape_quality": article.get("scrape_quality"),
        "char_count": article.get("char_count"),
        "scraped_at": article["scraped_at"],
    }
    return parsed


def extract_store_signals(
    client: Any,
    deployment: str,
    url: str,
    store_id: str = "",
    retailer: str = "",
    source_type: str = "retailer_page",
    skip_llm: bool = False,
) -> dict:
    """End-to-end scrape + optional LLM extraction for one URL."""
    article = scrape(url)

    if skip_llm:
        return {"scrape": article, "_meta": {"store_id": store_id, "source_type": source_type}}

    if article["scrape_quality"] == "blocked":
        return {
            "error": "Bot protection detected — page not scrapeable with simple HTTP",
            "scrape_preview": article["text"][:300],
            "_meta": {
                "store_id": store_id,
                "retailer": retailer,
                "source_type": source_type,
                "url": url,
                "scrape_quality": "blocked",
            },
        }

    analysis = analyze_scraped_page(
        client, deployment, article, store_id, retailer, source_type
    )
    return {**analysis, "scrape_quality": article["scrape_quality"]}


def extract_full_synthetic_profile(
    client: Any,
    deployment: str,
    retailer: str,
    store_id: str,
    retailer_scrape: dict,
    demographics_scrape: dict | None = None,
) -> dict:
    """
    One LLM call to map combined scraped text into synthetic-style tables.
    Returns parsed JSON matching FULL_PROFILE_SCHEMA in synthetic_mapper.
    """
    from src.synthetic_mapper import FULL_PROFILE_SCHEMA

    demo_text = ""
    if demographics_scrape:
        demo_text = f"\n\n--- demographics page ---\n{demographics_scrape.get('text', '')[:6000]}"

    prompt = f"""
You are converting scraped public web pages into Retail StoreDNA datasets.
The output must match the synthetic data model used for store analytics.

Retailer: {retailer}
Store ID: {store_id}

Rules:
- Return ONLY valid JSON matching the schema below.
- Use only facts present in the scraped content.
- customer_reviews: include only real review-like text found on pages. If none, return [].
- assortment_snapshot: infer categories/departments mentioned on the store page (Grocery, Dairy, etc.).
- product_descriptions: list representative products/departments/services described on the page.
- local_news: only if news/events are mentioned; otherwise [].
- operational_reports: only if operational issues/audits are mentioned; otherwise [].
- image_audits: if store/department images or zones are described, create audit rows; else [].
- Leave unknown numeric fields as empty strings.

JSON schema:
{FULL_PROFILE_SCHEMA}

Retailer page:
{retailer_scrape.get('text', '')[:9000]}
{demo_text}
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = _strip_json_fences(response.choices[0].message.content)
    return json.loads(content)
