"""
Store Pulse review insights: embeddings for theme matching + Azure OpenAI for ratings/summaries.

Results are cached under data/USA_100_Stores/enriched/review_pulse_insights.json
so the front-end API does not re-call the LLM on every restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.store_dna_enrichment import EnrichmentConfig, embed_texts

REVIEW_THEMES = (
    {
        "id": "service",
        "label": "Staff & service",
        "query": (
            "Customer review about store staff, service quality, employees, cashiers, "
            "managers, helpfulness, rudeness, friendliness, or customer support."
        ),
    },
    {
        "id": "availability",
        "label": "Product availability",
        "query": (
            "Customer review about product availability, out of stock items, shelves, "
            "inventory, missing products, order fulfillment, delivery, or pickup."
        ),
    },
    {
        "id": "value",
        "label": "Price & value",
        "query": (
            "Customer review about prices, pricing, affordability, expensive items, "
            "cheap deals, value for money, or cost."
        ),
    },
    {
        "id": "experience",
        "label": "Store experience",
        "query": (
            "Customer review about overall shopping experience, store cleanliness, wait times, "
            "checkout, parking, aisles, freshness, product quality, or atmosphere."
        ),
    },
)

_THEME_BY_ID = {t["id"]: t for t in REVIEW_THEMES}
_SIM_THRESHOLD = 0.22
_CACHE_NAME = "review_pulse_insights.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cache_path() -> Path:
    return _project_root() / "data" / "USA_100_Stores" / "enriched" / _CACHE_NAME


def _load_env() -> None:
    load_dotenv(_project_root() / ".env")


def _reviews_hash(group: pd.DataFrame) -> str:
    parts: list[str] = []
    for _, row in group.sort_values(by=[c for c in ("review_id", "review_date") if c in group.columns]).iterrows():
        parts.append(
            f"{row.get('review_id', '')}|{row.get('sentiment', '')}|{row.get('rating', '')}|{row.get('review_text', '')}"
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


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


def _clip(text: str, max_len: int = 350) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


def _empty_theme(theme: dict[str, str]) -> dict[str, Any]:
    return {
        "id": theme["id"],
        "label": theme["label"],
        "rating": None,
        "display": "",
        "mention_count": 0,
        "empty": True,
        "summary": "",
    }


def _empty_summaries() -> dict[str, Any]:
    return {
        "positive": {"count": 0, "summary": "", "excerpts": []},
        "negative": {"count": 0, "summary": "", "excerpts": []},
    }


def _load_cache() -> dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {"version": 2, "stores": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 2, "stores": {}}
        data.setdefault("version", 2)
        data.setdefault("stores", {})
        return data
    except (OSError, json.JSONDecodeError):
        return {"version": 2, "stores": {}}


def _save_cache(cache: dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _llm_store_insights(
    client: Any,
    deployment: str,
    store_id: str,
    theme_buckets: dict[str, list[str]],
    positive_texts: list[str],
    negative_texts: list[str],
    temperature: float = 0.2,
) -> dict[str, Any]:
    theme_blocks: list[str] = []
    for theme in REVIEW_THEMES:
        texts = theme_buckets.get(theme["id"], [])
        if not texts:
            theme_blocks.append(f"### {theme['id']} ({theme['label']})\n(no matching reviews)")
        else:
            lines = "\n".join(f"- {_clip(t, 280)}" for t in texts[:8])
            theme_blocks.append(f"### {theme['id']} ({theme['label']})\n{lines}")

    pos_block = "\n".join(f"- {_clip(t, 280)}" for t in positive_texts[:10]) or "(none)"
    neg_block = "\n".join(f"- {_clip(t, 280)}" for t in negative_texts[:10]) or "(none)"

    prompt = f"""You are a retail analyst summarizing customer reviews for store {store_id}.

For each theme below, if there are matching reviews:
- rating: number from 1.0 to 5.0 based only on those reviews
- summary: 2 short sentences synthesizing what customers said about that theme (do NOT paste review text verbatim)

If a theme has no matching reviews, set it to null.

Also write:
- positive_summary: one cohesive paragraph synthesizing ALL positive reviews (not bullet pasted quotes)
- negative_summary: one cohesive paragraph synthesizing ALL negative reviews (not bullet pasted quotes)
If there are no reviews for a polarity, use an empty string.

Return ONLY valid JSON with this shape:
{{
  "themes": {{
    "service": {{"rating": 4.0, "summary": "..."}},
    "availability": null,
    "value": {{"rating": 3.0, "summary": "..."}},
    "experience": {{"rating": 5.0, "summary": "..."}}
  }},
  "positive_summary": "...",
  "negative_summary": "..."
}}

Theme-matched reviews:
{chr(10).join(theme_blocks)}

Positive reviews:
{pos_block}

Negative reviews:
{neg_block}
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return _parse_json_content(response.choices[0].message.content or "{}")


def _assemble_payload(
    parsed: dict[str, Any],
    theme_buckets: dict[str, list[str]],
    positive_texts: list[str],
    negative_texts: list[str],
) -> dict[str, Any]:
    themes_parsed = parsed.get("themes") or {}
    metrics: list[dict[str, Any]] = []
    for theme in REVIEW_THEMES:
        mentions = theme_buckets.get(theme["id"], [])
        if not mentions:
            metrics.append(_empty_theme(theme))
            continue
        raw = themes_parsed.get(theme["id"])
        rating = None
        summary = ""
        if isinstance(raw, dict):
            try:
                rating = float(raw.get("rating"))
            except (TypeError, ValueError):
                rating = None
            summary = str(raw.get("summary") or "").strip()
        if rating is None or not (1.0 <= rating <= 5.0):
            rating = 3.0
            summary = summary or "Customers mentioned this theme; rating estimated as mid-scale."
        rating = round(rating, 1)
        metrics.append(
            {
                "id": theme["id"],
                "label": theme["label"],
                "rating": rating,
                "display": f"{rating:.1f} / 5",
                "mention_count": len(mentions),
                "empty": False,
                "summary": summary,
            }
        )

    pos_summary = str(parsed.get("positive_summary") or "").strip()
    neg_summary = str(parsed.get("negative_summary") or "").strip()
    if positive_texts and not pos_summary:
        pos_summary = "Positive customer themes were present, but a full summary could not be generated."
    if negative_texts and not neg_summary:
        neg_summary = "Negative customer themes were present, but a full summary could not be generated."
    return {
        "theme_ratings": metrics,
        "summaries": {
            "positive": {
                "count": len(positive_texts),
                "summary": pos_summary if positive_texts else "",
                "excerpts": [],
            },
            "negative": {
                "count": len(negative_texts),
                "summary": neg_summary if negative_texts else "",
                "excerpts": [],
            },
        },
    }


def build_review_insights(
    reviews: pd.DataFrame,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Return {store_id: {theme_ratings, summaries}} using embeddings + LLM.

    Falls back to empty themes if Azure OpenAI is unavailable.
    """
    if reviews.empty or "store_id" not in reviews.columns:
        return {}

    _load_env()
    force = force_refresh or os.getenv("REVIEW_INSIGHTS_FORCE_REFRESH", "").lower() in {"1", "true", "yes"}
    max_stores_env = os.getenv("REVIEW_INSIGHTS_MAX_STORES", "").strip()
    max_stores = int(max_stores_env) if max_stores_env.isdigit() else None

    cache = _load_cache()
    stores_cache: dict[str, Any] = cache["stores"]
    out: dict[str, dict[str, Any]] = {}

    store_ids = [str(s) for s in reviews["store_id"].dropna().unique().tolist()]
    store_ids.sort()
    if max_stores is not None:
        store_ids = store_ids[:max_stores]

    # Serve cache hits first
    pending: list[str] = []
    for store_id in store_ids:
        group = reviews[reviews["store_id"].astype(str) == store_id]
        digest = _reviews_hash(group)
        cached = stores_cache.get(store_id)
        if (
            not force
            and isinstance(cached, dict)
            and cached.get("reviews_hash") == digest
            and cached.get("theme_ratings")
            and cached.get("summaries")
        ):
            out[store_id] = {
                "theme_ratings": cached["theme_ratings"],
                "summaries": cached["summaries"],
            }
        else:
            pending.append(store_id)

    if not pending:
        return out

    try:
        config = EnrichmentConfig.from_env(_project_root())
        client = config.create_client()
    except Exception as exc:  # noqa: BLE001
        print(f"[review insights] Azure OpenAI unavailable ({exc}); leaving pending stores empty.")
        for store_id in pending:
            out[store_id] = {
                "theme_ratings": [_empty_theme(t) for t in REVIEW_THEMES],
                "summaries": _empty_summaries(),
            }
        return out

    # Embed theme queries once
    theme_vecs = embed_texts(
        client,
        [t["query"] for t in REVIEW_THEMES],
        config.embedding_deployment,
        batch_size=config.embed_batch_size,
        progress_label="theme queries",
    )

    # Embed only reviews for pending stores
    pending_mask = reviews["store_id"].astype(str).isin(pending)
    pending_reviews = reviews.loc[pending_mask].copy().reset_index(drop=True)
    texts = pending_reviews.get("review_text", pd.Series([""] * len(pending_reviews))).fillna("").astype(str).tolist()
    print(f"[review insights] Embedding {len(texts)} reviews for {len(pending)} stores…")
    review_vecs = embed_texts(
        client,
        texts,
        config.embedding_deployment,
        batch_size=config.embed_batch_size,
        progress_label="reviews",
    )
    sims = _cosine_matrix(review_vecs, theme_vecs)  # (n_reviews, n_themes)

    for idx, store_id in enumerate(pending, start=1):
        pos_indices = np.flatnonzero(pending_reviews["store_id"].astype(str).to_numpy() == store_id)
        store_rows = pending_reviews.iloc[pos_indices]
        store_sims = sims[pos_indices]

        theme_buckets: dict[str, list[str]] = {t["id"]: [] for t in REVIEW_THEMES}
        positive_texts: list[str] = []
        negative_texts: list[str] = []

        for row_i in range(len(store_rows)):
            row = store_rows.iloc[row_i]
            text = str(row.get("review_text", "") or "").strip()
            sentiment = str(row.get("sentiment", "neutral")).lower()
            if sentiment == "positive" and text:
                positive_texts.append(text)
            elif sentiment == "negative" and text:
                negative_texts.append(text)
            if not text:
                continue
            row_sim = store_sims[row_i]
            matched_any = False
            for theme_i, theme in enumerate(REVIEW_THEMES):
                if float(row_sim[theme_i]) >= _SIM_THRESHOLD:
                    theme_buckets[theme["id"]].append(text)
                    matched_any = True
            if not matched_any:
                best_i = int(np.argmax(row_sim))
                if float(row_sim[best_i]) >= 0.18:
                    theme_buckets[REVIEW_THEMES[best_i]["id"]].append(text)

        matched_total = sum(len(v) for v in theme_buckets.values())
        print(
            f"[review insights] LLM summarize {store_id} ({idx}/{len(pending)}) "
            f"theme_hits={matched_total} pos={len(positive_texts)} neg={len(negative_texts)}…"
        )
        try:
            parsed = _llm_store_insights(
                client,
                config.gpt_deployment,
                store_id,
                theme_buckets,
                positive_texts,
                negative_texts,
                temperature=config.gpt_temperature,
            )
            payload = _assemble_payload(parsed, theme_buckets, positive_texts, negative_texts)
        except Exception as exc:  # noqa: BLE001
            print(f"[review insights] LLM failed for {store_id}: {exc}")
            theme_ratings = []
            for theme in REVIEW_THEMES:
                mentions = theme_buckets[theme["id"]]
                if not mentions:
                    theme_ratings.append(_empty_theme(theme))
                else:
                    theme_ratings.append(
                        {
                            **_empty_theme(theme),
                            "empty": False,
                            "rating": 3.0,
                            "display": "3.0 / 5",
                            "mention_count": len(mentions),
                            "summary": "Summary unavailable.",
                        }
                    )
            payload = {
                "theme_ratings": theme_ratings,
                "summaries": {
                    "positive": {
                        "count": len(positive_texts),
                        "summary": "" if not positive_texts else "Positive summary unavailable.",
                        "excerpts": [],
                    },
                    "negative": {
                        "count": len(negative_texts),
                        "summary": "" if not negative_texts else "Negative summary unavailable.",
                        "excerpts": [],
                    },
                },
            }

        digest = _reviews_hash(reviews[reviews["store_id"].astype(str) == store_id])
        stores_cache[store_id] = {
            "reviews_hash": digest,
            "theme_ratings": payload["theme_ratings"],
            "summaries": payload["summaries"],
            "updated_at": time.time(),
        }
        out[store_id] = payload
        _save_cache({"version": 2, "stores": stores_cache})
        time.sleep(0.25)

    return out
