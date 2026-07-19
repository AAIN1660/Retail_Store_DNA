"""
LLM shared learnings for similar-store Face-Off.

Uses embedding similarity + store KPI / review / news data (not hardcoded tips).
Cached under data/USA_100_Stores/enriched/peer_learnings_cache.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.store_dna_enrichment import EnrichmentConfig

_CACHE_NAME = "peer_learnings_cache.json"
_CACHE_VERSION = 2
_STATUS_RANK = {"good": 2, "stable": 1, "attention": 0}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cache_path() -> Path:
    return _project_root() / "data" / "USA_100_Stores" / "enriched" / _CACHE_NAME


def _load_env() -> None:
    load_dotenv(_project_root() / ".env")


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


def _load_cache() -> dict[str, Any]:
    path = _cache_path()
    empty = {"version": _CACHE_VERSION, "pairs": {}}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return empty
        # Drop old long-paragraph cache entries.
        if data.get("version") != _CACHE_VERSION:
            return empty
        data.setdefault("version", _CACHE_VERSION)
        data.setdefault("pairs", {})
        return data
    except (OSError, json.JSONDecodeError):
        return empty


def _save_cache(cache: dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cache["version"] = _CACHE_VERSION
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _clip_words(text: str, max_words: int = 14) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words).strip()
    return " ".join(words[:max_words]).rstrip(".,;:") + "…"


def _extract_bullets(item: dict[str, Any], gap: dict[str, Any]) -> list[str]:
    raw = item.get("bullets")
    bullets: list[str] = []
    if isinstance(raw, list):
        for entry in raw:
            text = _clip_words(str(entry).strip(), 16)
            if text:
                bullets.append(text)
    if bullets:
        return bullets[:3]

    summary = str(item.get("summary") or "").strip()
    if summary:
        parts = re.split(r"(?<=[.!?])\s+", summary)
        for part in parts:
            text = _clip_words(part.strip(), 16)
            if text:
                bullets.append(text)
            if len(bullets) >= 3:
                break
    if bullets:
        return bullets

    direction = "higher" if gap.get("higher_is_better") else "lower"
    return [
        f"{gap['leader_store_name']} leads ({gap['leader_display']} vs {gap['follower_display']})",
        f"Copy the practices that keep this metric {direction}",
        f"Apply them at {gap['follower_store_name']} to close the gap",
    ]


def _pair_key(store_a: str, store_b: str) -> str:
    left, right = sorted([store_a, store_b])
    return f"{left}__{right}"


def _store_fingerprint(store: dict[str, Any]) -> str:
    pulse = store.get("business", {}).get("store_pulse", {})
    payload = {
        "status_tone": store.get("executive", {}).get("status_tone"),
        "metrics_above_avg": store.get("executive", {}).get("metrics_above_avg"),
        "kpis": store.get("executive", {}).get("kpis"),
        "theme_ratings": pulse.get("reviews", {}).get("theme_ratings"),
        "review_summaries": pulse.get("reviews", {}).get("summaries"),
        "news_overview": pulse.get("news", {}).get("overview"),
        "news_headlines": [
            {"headline": h.get("headline"), "summary": h.get("summary")}
            for h in (pulse.get("news", {}).get("headlines") or [])[:4]
        ],
        "focus_areas": store.get("business", {}).get("focus_areas"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _context_hash(store_a: dict[str, Any], store_b: dict[str, Any], similarity: float | None) -> str:
    sim = f"{similarity:.4f}" if similarity is not None else "na"
    blob = f"{_store_fingerprint(store_a)}|{_store_fingerprint(store_b)}|{sim}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _kpi_winner(kpi_a: dict[str, Any], kpi_b: dict[str, Any]) -> str:
    if kpi_a.get("value") == kpi_b.get("value"):
        return "tie"
    higher_is_better = bool(kpi_a.get("higher_is_better", True))
    a_wins = kpi_a["value"] > kpi_b["value"] if higher_is_better else kpi_a["value"] < kpi_b["value"]
    return "a" if a_wins else "b"


def _compact_store(store: dict[str, Any]) -> dict[str, Any]:
    pulse = store.get("business", {}).get("store_pulse", {})
    reviews = pulse.get("reviews", {})
    news = pulse.get("news", {})
    themes = []
    for theme in reviews.get("theme_ratings") or []:
        if theme.get("empty"):
            continue
        themes.append(
            {
                "label": theme.get("label"),
                "rating": theme.get("display"),
                "summary": (theme.get("summary") or "")[:220],
            }
        )
    headlines = []
    for item in (news.get("headlines") or [])[:4]:
        headlines.append(
            {
                "headline": item.get("headline"),
                "summary": (item.get("summary") or "")[:180],
                "demand_impact": item.get("demand_impact"),
            }
        )
    return {
        "store_id": store.get("store_id"),
        "store_name": store.get("store_name"),
        "retailer": store.get("retailer"),
        "city": store.get("city"),
        "state": store.get("state"),
        "format": store.get("format"),
        "status": store.get("executive", {}).get("status"),
        "status_tone": store.get("executive", {}).get("status_tone"),
        "metrics_above_avg": store.get("executive", {}).get("metrics_above_avg"),
        "kpis": [
            {
                "id": k.get("id"),
                "label": k.get("label"),
                "value": k.get("value"),
                "display": k.get("display"),
                "network_avg_display": k.get("network_avg_display"),
                "vs_network": k.get("vs_network"),
                "higher_is_better": k.get("higher_is_better"),
            }
            for k in store.get("executive", {}).get("kpis") or []
        ],
        "review_themes": themes,
        "positive_review_summary": (reviews.get("summaries") or {}).get("positive", {}).get("summary", "")[:280],
        "negative_review_summary": (reviews.get("summaries") or {}).get("negative", {}).get("summary", "")[:280],
        "news_overview": (news.get("overview") or "")[:280],
        "news_headlines": headlines,
        "focus_areas": store.get("business", {}).get("focus_areas") or [],
    }


def _metric_gaps(store_a: dict[str, Any], store_b: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    kpis_b = {k["id"]: k for k in store_b.get("executive", {}).get("kpis") or []}
    for kpi_a in store_a.get("executive", {}).get("kpis") or []:
        kpi_b = kpis_b.get(kpi_a["id"])
        if not kpi_b:
            continue
        winner = _kpi_winner(kpi_a, kpi_b)
        if winner == "tie":
            continue
        leader = store_a if winner == "a" else store_b
        follower = store_b if winner == "a" else store_a
        leader_kpi = kpi_a if winner == "a" else kpi_b
        follower_kpi = kpi_b if winner == "a" else kpi_a
        gaps.append(
            {
                "kpi_id": kpi_a["id"],
                "label": kpi_a["label"],
                "leader_store_id": leader["store_id"],
                "leader_store_name": leader["store_name"],
                "follower_store_id": follower["store_id"],
                "follower_store_name": follower["store_name"],
                "leader_display": leader_kpi.get("display"),
                "follower_display": follower_kpi.get("display"),
                "leader_vs_network": leader_kpi.get("vs_network"),
                "follower_vs_network": follower_kpi.get("vs_network"),
                "higher_is_better": leader_kpi.get("higher_is_better"),
            }
        )
    return gaps


def should_generate_learnings(store_a: dict[str, Any], store_b: dict[str, Any]) -> bool:
    tone_a = store_a.get("executive", {}).get("status_tone")
    tone_b = store_b.get("executive", {}).get("status_tone")
    return bool(tone_a and tone_b and tone_a != tone_b)


def _stronger_store(store_a: dict[str, Any], store_b: dict[str, Any]) -> dict[str, Any] | None:
    rank_a = _STATUS_RANK.get(store_a.get("executive", {}).get("status_tone", ""), 0)
    rank_b = _STATUS_RANK.get(store_b.get("executive", {}).get("status_tone", ""), 0)
    if rank_a == rank_b:
        above_a = store_a.get("executive", {}).get("metrics_above_avg") or 0
        above_b = store_b.get("executive", {}).get("metrics_above_avg") or 0
        if above_a == above_b:
            return None
        return store_a if above_a > above_b else store_b
    return store_a if rank_a > rank_b else store_b


def _llm_peer_learnings(
    client: Any,
    deployment: str,
    store_a: dict[str, Any],
    store_b: dict[str, Any],
    similarity: float | None,
    gaps: list[dict[str, Any]],
    temperature: float = 0.2,
) -> dict[str, Any]:
    stronger = _stronger_store(store_a, store_b)
    weaker = store_b if stronger and stronger["store_id"] == store_a["store_id"] else store_a
    if stronger is None:
        stronger_name = "the healthier store"
        weaker_name = "the trailing store"
    else:
        stronger_name = stronger["store_name"]
        weaker_name = weaker["store_name"]

    sim_pct = f"{similarity * 100:.0f}%" if similarity is not None else "unknown"
    prompt = f"""You are a retail operations coach. Write SHORT, scannable transfer tips.

Two stores are similar by Store DNA embeddings (playbook match {sim_pct}).
They are in DIFFERENT health categories, so the stronger store should share concrete learnings
with the weaker store on metrics where one is ahead.

Store A context JSON:
{json.dumps(_compact_store(store_a), indent=2)}

Store B context JSON:
{json.dumps(_compact_store(store_b), indent=2)}

Metric gaps where one store is ahead:
{json.dumps(gaps, indent=2)}

Overall healthier store (category): {stronger_name}
Trailing store (category): {weaker_name}

For EACH metric gap, write one learning object:
- kpi_id: must match the gap kpi_id
- leader_store_id / follower_store_id: who is ahead / behind on THAT metric
- headline: max 8 words (action-focused, e.g. "Copy Dallas pick accuracy habits")
- bullets: exactly 2 short tips. Each tip max 12 words. No paragraphs.
  Include the key number gap in one bullet. Ground tips in review themes / news / focus areas when present.
  Do NOT invent KPIs. Do NOT write long sentences.

Also write:
- overview: ONE short line (max 12 words) on why this pair is a useful coaching match.

Return ONLY valid JSON:
{{
  "overview": "Similar ops profile; clear coaching gaps",
  "learnings": [
    {{
      "kpi_id": "fulfillment",
      "leader_store_id": "USR-001",
      "follower_store_id": "USR-008",
      "headline": "Copy leader fulfillment habits",
      "bullets": [
        "Leader is ahead by 0.2 pts",
        "Mirror their order accuracy checklist"
      ]
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return _parse_json_content(response.choices[0].message.content or "{}")


def _normalize_learnings(
    parsed: dict[str, Any],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {g["kpi_id"]: g for g in gaps}
    items: list[dict[str, Any]] = []
    raw = parsed.get("learnings") or []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            kpi_id = str(item.get("kpi_id") or "")
            if kpi_id not in by_id:
                continue
            gap = by_id[kpi_id]
            bullets = _extract_bullets(item, gap)
            headline = _clip_words(
                str(item.get("headline") or "").strip() or f"Copy from {gap['leader_store_name']}",
                8,
            )
            items.append(
                {
                    "kpi_id": kpi_id,
                    "label": gap["label"],
                    "leader_store_id": gap["leader_store_id"],
                    "follower_store_id": gap["follower_store_id"],
                    "leader_store_name": gap["leader_store_name"],
                    "follower_store_name": gap["follower_store_name"],
                    "headline": headline,
                    "bullets": bullets,
                    "summary": " · ".join(bullets),
                }
            )
    # Ensure every gap has something if LLM skipped one
    covered = {i["kpi_id"] for i in items}
    for gap in gaps:
        if gap["kpi_id"] in covered:
            continue
        bullets = _extract_bullets({}, gap)
        items.append(
            {
                "kpi_id": gap["kpi_id"],
                "label": gap["label"],
                "leader_store_id": gap["leader_store_id"],
                "follower_store_id": gap["follower_store_id"],
                "leader_store_name": gap["leader_store_name"],
                "follower_store_name": gap["follower_store_name"],
                "headline": f"Copy from {gap['leader_store_name']}",
                "bullets": bullets,
                "summary": " · ".join(bullets),
            }
        )
    overview = _clip_words(str(parsed.get("overview") or "").strip(), 12)
    return {
        "overview": overview,
        "learnings": items,
        "source": "azure_openai",
    }


def build_peer_learnings(
    store_a: dict[str, Any],
    store_b: dict[str, Any],
    similarity: float | None,
    *,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    """
    Return LLM learnings for a similar pair when status categories differ.
    Returns None when learnings should not show.
    """
    if not store_a or not store_b:
        return None
    if not should_generate_learnings(store_a, store_b):
        return None

    gaps = _metric_gaps(store_a, store_b)
    if not gaps:
        return {
            "overview": "Similar profile; no clear transfer tip",
            "learnings": [],
            "source": "none",
        }

    _load_env()
    force = force_refresh or os.getenv("PEER_LEARNINGS_FORCE_REFRESH", "").lower() in {"1", "true", "yes"}
    key = _pair_key(store_a["store_id"], store_b["store_id"])
    digest = _context_hash(store_a, store_b, similarity)
    cache = _load_cache()
    cached = cache["pairs"].get(key)
    if (
        not force
        and isinstance(cached, dict)
        and cached.get("context_hash") == digest
        and isinstance(cached.get("learnings"), list)
    ):
        return {
            "overview": cached.get("overview") or "",
            "learnings": cached.get("learnings") or [],
            "source": cached.get("source") or "cache",
        }

    try:
        config = EnrichmentConfig.from_env(_project_root())
        client = config.create_client()
        parsed = _llm_peer_learnings(
            client,
            config.gpt_deployment,
            store_a,
            store_b,
            similarity,
            gaps,
            temperature=config.gpt_temperature,
        )
        payload = _normalize_learnings(parsed, gaps)
    except Exception as exc:  # noqa: BLE001
        print(f"[peer learnings] LLM failed for {key}: {exc}")
        payload = _normalize_learnings({"overview": "", "learnings": []}, gaps)
        payload["source"] = "fallback"
        payload["overview"] = "Peer match found; using short data-backed tips"

    cache["pairs"][key] = {
        "context_hash": digest,
        "overview": payload["overview"],
        "learnings": payload["learnings"],
        "source": payload["source"],
        "similarity": similarity,
        "updated_at": time.time(),
    }
    _save_cache(cache)
    return payload
