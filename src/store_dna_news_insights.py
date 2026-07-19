"""
Store Pulse local-news insights: Azure OpenAI picks main headlines and writes summaries.

Cached under data/USA_100_Stores/enriched/news_pulse_insights.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.store_dna_enrichment import EnrichmentConfig

_CACHE_NAME = "news_pulse_insights.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cache_path() -> Path:
    return _project_root() / "data" / "USA_100_Stores" / "enriched" / _CACHE_NAME


def _load_env() -> None:
    load_dotenv(_project_root() / ".env")


def _news_hash(group: pd.DataFrame) -> str:
    parts: list[str] = []
    sort_cols = [c for c in ("published_date", "news_id") if c in group.columns]
    ordered = group.sort_values(by=sort_cols) if sort_cols else group
    for _, row in ordered.iterrows():
        parts.append(
            f"{row.get('news_id', '')}|{row.get('headline', '')}|{row.get('summary', '')}|"
            f"{row.get('event_type', '')}|{row.get('demand_impact', '')}"
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _parse_json_content(content: str) -> dict[str, Any]:
    text = (content or "{}").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _clip(text: str, max_len: int = 280) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


def _empty_news_insights() -> dict[str, Any]:
    return {"article_count": 0, "overview": "", "headlines": []}


def _load_cache() -> dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {"version": 1, "stores": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 1, "stores": {}}
        data.setdefault("version", 1)
        data.setdefault("stores", {})
        return data
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "stores": {}}


def _save_cache(cache: dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _candidate_articles(group: pd.DataFrame, limit: int = 12) -> list[dict[str, str]]:
    sort_cols = [c for c in ("published_date",) if c in group.columns]
    ordered = group.sort_values(by=sort_cols, ascending=False) if sort_cols else group
    articles: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in ordered.iterrows():
        headline = " ".join(str(row.get("headline", "") or "").split())
        if not headline:
            continue
        key = headline.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        articles.append(
            {
                "headline": headline,
                "summary": " ".join(str(row.get("summary", "") or "").split()),
                "event_type": str(row.get("event_type", "") or ""),
                "demand_impact": str(row.get("demand_impact", "") or ""),
                "published_date": str(row.get("published_date", "") or "")[:10],
                "source": str(row.get("source", "") or ""),
            }
        )
        if len(articles) >= limit:
            break
    return articles


def _llm_news_insights(
    client: Any,
    deployment: str,
    store_id: str,
    city: str,
    state: str,
    articles: list[dict[str, str]],
    temperature: float = 0.2,
) -> dict[str, Any]:
    lines = []
    for i, article in enumerate(articles, start=1):
        lines.append(
            f"{i}. [{article['published_date']}] ({article['event_type']}/{article['demand_impact']}) "
            f"{article['headline']}\n   note: {_clip(article['summary'] or article['headline'], 220)}"
        )
    joined = "\n".join(lines)

    prompt = f"""You are a retail analyst summarizing local/store-relevant news for store {store_id}
in {city}, {state}.

From the articles below:
1) Write overview: 2 sentences on what matters most for this store right now.
2) Choose the 3 to 5 MOST important distinct stories (skip near-duplicates).
3) For each chosen story, write:
   - headline: a clean, concise headline (you may tighten the original)
   - summary: 2 short sentences in your own words (do NOT paste the source text)
   - why_it_matters: one sentence on store impact
   - event_type and demand_impact from the source article when possible

Return ONLY valid JSON:
{{
  "overview": "...",
  "headlines": [
    {{
      "headline": "...",
      "summary": "...",
      "why_it_matters": "...",
      "event_type": "local_news",
      "demand_impact": "neutral",
      "published_date": "YYYY-MM-DD"
    }}
  ]
}}

Articles:
{joined}
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return _parse_json_content(response.choices[0].message.content or "{}")


def _normalize_payload(parsed: dict[str, Any], article_count: int) -> dict[str, Any]:
    overview = str(parsed.get("overview") or "").strip()
    raw_items = parsed.get("headlines") or []
    headlines: list[dict[str, str]] = []
    if isinstance(raw_items, list):
        for item in raw_items[:5]:
            if not isinstance(item, dict):
                continue
            headline = str(item.get("headline") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if not headline or not summary:
                continue
            headlines.append(
                {
                    "headline": headline,
                    "summary": summary,
                    "why_it_matters": str(item.get("why_it_matters") or "").strip(),
                    "event_type": str(item.get("event_type") or "").strip(),
                    "demand_impact": str(item.get("demand_impact") or "").strip(),
                    "published_date": str(item.get("published_date") or "").strip()[:10],
                }
            )
    return {
        "article_count": article_count,
        "overview": overview,
        "headlines": headlines,
    }


def build_news_insights(
    news: pd.DataFrame,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return {store_id: {article_count, overview, headlines}} via Azure OpenAI."""
    if news.empty or "store_id" not in news.columns:
        return {}

    _load_env()
    force = force_refresh or os.getenv("NEWS_INSIGHTS_FORCE_REFRESH", "").lower() in {"1", "true", "yes"}
    max_stores_env = os.getenv("NEWS_INSIGHTS_MAX_STORES", "").strip()
    max_stores = int(max_stores_env) if max_stores_env.isdigit() else None

    cache = _load_cache()
    stores_cache: dict[str, Any] = cache["stores"]
    out: dict[str, dict[str, Any]] = {}

    store_ids = [str(s) for s in news["store_id"].dropna().unique().tolist()]
    store_ids.sort()
    if max_stores is not None:
        store_ids = store_ids[:max_stores]

    pending: list[str] = []
    for store_id in store_ids:
        group = news[news["store_id"].astype(str) == store_id]
        digest = _news_hash(group)
        cached = stores_cache.get(store_id)
        if (
            not force
            and isinstance(cached, dict)
            and cached.get("news_hash") == digest
            and isinstance(cached.get("headlines"), list)
        ):
            out[store_id] = {
                "article_count": int(cached.get("article_count") or len(group)),
                "overview": str(cached.get("overview") or ""),
                "headlines": cached.get("headlines") or [],
            }
        else:
            pending.append(store_id)

    if not pending:
        return out

    try:
        config = EnrichmentConfig.from_env(_project_root())
        client = config.create_client()
    except Exception as exc:  # noqa: BLE001
        print(f"[news insights] Azure OpenAI unavailable ({exc}); leaving pending stores empty.")
        for store_id in pending:
            group = news[news["store_id"].astype(str) == store_id]
            out[store_id] = {
                "article_count": int(len(group)),
                "overview": "",
                "headlines": [],
            }
        return out

    for idx, store_id in enumerate(pending, start=1):
        group = news[news["store_id"].astype(str) == store_id]
        articles = _candidate_articles(group, limit=12)
        city = str(group["city"].iloc[0]) if "city" in group.columns and len(group) else ""
        state = str(group["state"].iloc[0]) if "state" in group.columns and len(group) else ""

        print(f"[news insights] LLM summarize {store_id} ({idx}/{len(pending)}) articles={len(articles)}…")
        if not articles:
            payload = _empty_news_insights()
        else:
            try:
                parsed = _llm_news_insights(
                    client,
                    config.gpt_deployment,
                    store_id,
                    city,
                    state,
                    articles,
                    temperature=config.gpt_temperature,
                )
                payload = _normalize_payload(parsed, article_count=int(len(group)))
                if not payload["headlines"]:
                    # Fallback: top 3 cleaned source headlines with short summaries
                    payload["headlines"] = [
                        {
                            "headline": a["headline"],
                            "summary": _clip(a["summary"] or a["headline"], 220),
                            "why_it_matters": "",
                            "event_type": a["event_type"],
                            "demand_impact": a["demand_impact"],
                            "published_date": a["published_date"],
                        }
                        for a in articles[:3]
                    ]
                    payload["overview"] = payload["overview"] or (
                        f"{len(group)} recent news items around {city or store_id}."
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[news insights] LLM failed for {store_id}: {exc}")
                payload = {
                    "article_count": int(len(group)),
                    "overview": "News summary unavailable.",
                    "headlines": [
                        {
                            "headline": a["headline"],
                            "summary": _clip(a["summary"] or a["headline"], 220),
                            "why_it_matters": "",
                            "event_type": a["event_type"],
                            "demand_impact": a["demand_impact"],
                            "published_date": a["published_date"],
                        }
                        for a in articles[:3]
                    ],
                }

        stores_cache[store_id] = {
            "news_hash": _news_hash(group),
            "article_count": payload["article_count"],
            "overview": payload["overview"],
            "headlines": payload["headlines"],
            "updated_at": time.time(),
        }
        out[store_id] = payload
        _save_cache({"version": 1, "stores": stores_cache})
        time.sleep(0.25)

    return out
