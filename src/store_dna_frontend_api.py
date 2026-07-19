"""
Front-end API service.

- KPIs / Store Pulse: curated + enriched CSV tables (ops, reviews, news)
- Peer similarity: Stage 6 Azure store embeddings (store-dna-store-vectors)

Hybrid vectors are not used.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

from src.store_dna_news_insights import build_news_insights
from src.store_dna_peer_learnings import build_peer_learnings
from src.store_dna_review_insights import REVIEW_THEMES, build_review_insights
from src.store_dna_vector_index import StoreVectorIndexConfig

STATUS_LABELS = {
    "good": "Doing well",
    "stable": "Steady",
    "attention": "Needs focus",
}

EXECUTIVE_KPI_DEFS = (
    {
        "id": "fulfillment",
        "label": "Fulfillment rate",
        "key": "fulfillment_rate",
        "format": "pct",
        "higher_is_better": True,
    },
    {
        "id": "oos",
        "label": "Out-of-stock rate",
        "key": "oos_rate",
        "format": "pct",
        "higher_is_better": False,
    },
    {
        "id": "shrink",
        "label": "Shrink rate",
        "key": "shrink_pct",
        "format": "pct",
        "higher_is_better": False,
    },
    {
        "id": "sentiment",
        "label": "Positive reviews",
        "key": "positive_review_pct",
        "format": "pct",
        "higher_is_better": True,
    },
    {
        "id": "complaints",
        "label": "Customer complaints",
        "key": "complaints_4w",
        "format": "count",
        "higher_is_better": False,
    },
)


@dataclass
class ApiCache:
    loaded_at: float
    summary: dict[str, Any]
    stores: list[dict[str, Any]]
    vectors: dict[str, np.ndarray]


_CACHE: ApiCache | None = None
_CACHE_TTL_SEC = 300


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env() -> None:
    load_dotenv(_project_root() / ".env")


def _data_root() -> Path:
    return _project_root() / "data" / "USA_100_Stores"


def _store_client() -> SearchClient:
    _load_env()
    config = StoreVectorIndexConfig.from_env(_project_root())
    return SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.search_api_key),
    )


def _paginate_search(client: SearchClient, select: list[str]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    skip = 0
    page_size = 1000
    while True:
        results = client.search(
            search_text="*",
            select=select,
            top=page_size,
            skip=skip,
        )
        batch = list(results)
        if not batch:
            break
        docs.extend(batch)
        skip += len(batch)
        if len(batch) < page_size:
            break
    return docs


def _kpi_status(value: float, network_avg: float, higher_is_better: bool) -> str:
    if higher_is_better:
        return "good" if value >= network_avg else "attention"
    return "good" if value <= network_avg else "attention"


def _vs_network(value: float, network_avg: float, higher_is_better: bool) -> str:
    if abs(value - network_avg) < 1e-9:
        return "on par"
    if higher_is_better:
        return "above" if value > network_avg else "below"
    return "below" if value < network_avg else "above"


def _format_value(value: float, fmt: str) -> str:
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    return f"{int(round(value))}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _pulse_kpi(
    label: str,
    value: float,
    network_avg: float,
    *,
    fmt: str = "count",
    higher_is_better: bool = True,
    display: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "value": round(value, 4) if fmt == "pct" else float(value),
        "display": display if display is not None else _format_value(value, fmt),
        "network_avg_display": _format_value(network_avg, fmt),
        "status": _kpi_status(value, network_avg, higher_is_better),
        "vs_network": _vs_network(value, network_avg, higher_is_better),
        "higher_is_better": higher_is_better,
    }


def _load_ops_metrics(ops: pd.DataFrame) -> pd.DataFrame:
    ops = ops.copy()
    ops["week_end_date"] = pd.to_datetime(ops["week_end_date"])
    latest = ops.groupby("store_id")["week_end_date"].transform("max")
    recent = ops[ops["week_end_date"] >= latest - pd.Timedelta(weeks=3)]
    metrics = recent.groupby("store_id").agg(
        fulfillment_rate=("fulfillment_rate", "mean"),
        oos_rate=("oos_rate", "mean"),
        shrink_pct=("shrink_pct", "mean"),
        complaints_4w=("customer_complaints", "sum"),
        labor_hours_4w=("labor_hours", "mean"),
    )
    return metrics.fillna(0.0)


def _load_review_metrics(reviews: pd.DataFrame) -> pd.DataFrame:
    if "sentiment" not in reviews.columns:
        reviews = reviews.copy()
        reviews["sentiment"] = "neutral"
    counts = reviews.groupby(["store_id", "sentiment"]).size().unstack(fill_value=0)
    for col in ("positive", "neutral", "negative"):
        if col not in counts.columns:
            counts[col] = 0
    counts["review_count"] = counts[["positive", "neutral", "negative"]].sum(axis=1)
    total = counts["review_count"].replace(0, np.nan)
    counts["positive_review_pct"] = (counts["positive"] / total).fillna(0.0)
    counts["negative_review_pct"] = (counts["negative"] / total).fillna(0.0)
    return counts[["review_count", "positive", "neutral", "negative", "positive_review_pct", "negative_review_pct"]]


def _load_reviews_frame() -> pd.DataFrame:
    root = _data_root()
    reviews_path = root / "enriched" / "reviews_enriched.csv"
    curated_path = root / "curated" / "stg_reviews.csv"
    path = reviews_path if reviews_path.exists() else curated_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_store_review_insights(reviews: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Embeddings match themes; Azure OpenAI rates themes and writes summaries."""
    return build_review_insights(reviews)


def _load_news_metrics(news: pd.DataFrame) -> pd.DataFrame:
    news_count = news.groupby("store_id").size().rename("news_count")
    if "event_type" in news.columns:
        local = news.groupby("store_id")["event_type"].apply(lambda s: int((s == "local_news").sum()))
        competitor = news.groupby("store_id")["event_type"].apply(
            lambda s: int((s == "new_competitor").sum())
        )
    else:
        local = news_count * 0
        competitor = news_count * 0
    if "demand_impact" in news.columns:
        negative = news.groupby("store_id")["demand_impact"].apply(lambda s: int((s == "negative").sum()))
    else:
        negative = news_count * 0
    return pd.DataFrame(
        {
            "news_count": news_count,
            "local_news_count": local,
            "competitor_mention_count": competitor,
            "negative_news_count": negative,
        }
    ).fillna(0.0)


def _load_curated_kpi_table() -> pd.DataFrame:
    """Build one KPI/pulse row per store from curated + enriched CSVs."""
    root = _data_root()
    curated = root / "curated"
    enriched = root / "enriched"

    dim = pd.read_csv(curated / "dim_store.csv")
    ops = pd.read_csv(curated / "fact_operations_weekly.csv")
    reviews_path = enriched / "reviews_enriched.csv"
    news_path = enriched / "news_enriched.csv"
    reviews = pd.read_csv(reviews_path if reviews_path.exists() else curated / "stg_reviews.csv")
    news = pd.read_csv(news_path if news_path.exists() else curated / "stg_news.csv")

    frame = dim.set_index("store_id")
    frame = frame.join(_load_ops_metrics(ops), how="left")
    frame = frame.join(_load_review_metrics(reviews), how="left")
    frame = frame.join(_load_news_metrics(news), how="left")
    return frame.fillna(0.0).reset_index()


def _load_store_embeddings() -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Load Stage 6 store embeddings for peer similarity."""
    client = _store_client()
    select = [
        "store_id",
        "retailer",
        "banner",
        "store_name",
        "city",
        "state",
        "store_format",
        "store_dna_vector",
    ]
    docs = _paginate_search(client, select=select)
    vectors: dict[str, np.ndarray] = {}
    for doc in docs:
        store_id = str(doc["store_id"])
        vector = np.asarray(doc.get("store_dna_vector") or [], dtype=np.float32)
        if vector.size:
            vectors[store_id] = vector
    return docs, vectors


def _build_executive(kpis: dict[str, float], network: dict[str, float]) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    above = 0
    for definition in EXECUTIVE_KPI_DEFS:
        key = definition["key"]
        value = float(kpis.get(key, 0.0))
        avg = float(network.get(key, 0.0))
        status = _kpi_status(value, avg, definition["higher_is_better"])
        vs = _vs_network(value, avg, definition["higher_is_better"])
        is_good = (definition["higher_is_better"] and vs in ("above", "on par")) or (
            not definition["higher_is_better"] and vs in ("below", "on par")
        )
        if is_good:
            above += 1
        cards.append(
            {
                "id": definition["id"],
                "label": definition["label"],
                "value": value,
                "display": _format_value(value, definition["format"]),
                "network_avg_display": _format_value(avg, definition["format"]),
                "status": status,
                "vs_network": vs,
                "higher_is_better": definition["higher_is_better"],
            }
        )

    if above >= 4:
        tone = "good"
    elif above >= 2:
        tone = "stable"
    else:
        tone = "attention"

    return {
        "status": STATUS_LABELS[tone],
        "status_tone": tone,
        "headline": f"{above} of 5 metrics are at or better than the network average.",
        "metrics_above_avg": above,
        "kpis": cards,
    }


def _empty_review_insights() -> dict[str, Any]:
    return {
        "theme_ratings": [
            {
                "id": theme["id"],
                "label": theme["label"],
                "rating": None,
                "display": "",
                "mention_count": 0,
                "empty": True,
                "summary": "",
            }
            for theme in REVIEW_THEMES
        ],
        "summaries": {
            "positive": {"count": 0, "summary": "", "excerpts": []},
            "negative": {"count": 0, "summary": "", "excerpts": []},
        },
    }


def _load_news_frame() -> pd.DataFrame:
    root = _data_root()
    enriched = root / "enriched" / "news_enriched.csv"
    curated = root / "curated" / "stg_news.csv"
    path = enriched if enriched.exists() else curated
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _empty_news_insights() -> dict[str, Any]:
    return {"article_count": 0, "overview": "", "headlines": []}


def _build_pulse(
    kpis: dict[str, float],
    network: dict[str, float],
    review_insights: dict[str, Any] | None = None,
    news_insights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    review_count = float(kpis.get("review_count", 0))
    positive = int(round(float(kpis.get("positive", review_count * float(kpis.get("positive_review_pct", 0))))))
    negative = int(round(float(kpis.get("negative", review_count * float(kpis.get("negative_review_pct", 0))))))
    neutral = int(round(float(kpis.get("neutral", max(review_count - positive - negative, 0)))))
    pos_pct = float(kpis.get("positive_review_pct", 0))
    neg_pct = float(kpis.get("negative_review_pct", 0))
    insights = review_insights or _empty_review_insights()
    news = news_insights or _empty_news_insights()
    article_count = int(news.get("article_count") or kpis.get("news_count", 0))

    return {
        "reviews": {
            "kpis": [
                _pulse_kpi("Total reviews", review_count, network.get("review_count", 0), higher_is_better=True),
                _pulse_kpi(
                    "Positive reviews",
                    positive,
                    network.get("positive", 0),
                    higher_is_better=True,
                    display=f"{positive} ({pos_pct * 100:.0f}%)" if review_count else "0",
                ),
                _pulse_kpi(
                    "Negative reviews",
                    negative,
                    network.get("negative", 0),
                    higher_is_better=False,
                    display=f"{negative} ({neg_pct * 100:.0f}%)" if review_count else "0",
                ),
            ],
            "theme_ratings": insights["theme_ratings"],
            "summaries": insights["summaries"],
            "breakdown": {
                "review_count": int(review_count),
                "positive": positive,
                "neutral": neutral,
                "negative": negative,
                "positive_share_display": f"{pos_pct * 100:.1f}%",
                "negative_share_display": f"{neg_pct * 100:.1f}%",
            },
        },
        "news": {
            "article_count": article_count,
            "overview": str(news.get("overview") or ""),
            "headlines": news.get("headlines") or [],
        },
    }


def _focus_areas(executive: dict[str, Any]) -> list[dict[str, str]]:
    areas: list[dict[str, str]] = []
    for kpi in executive["kpis"]:
        if kpi["status"] != "attention":
            continue
        detail = (
            f"Below network average ({kpi['display']} vs {kpi['network_avg_display']})."
            if kpi["higher_is_better"]
            else f"Above network average ({kpi['display']} vs {kpi['network_avg_display']})."
        )
        areas.append({"title": kpi["label"], "detail": detail})
    return areas[:3]


def _top_peers(
    store_id: str,
    vectors: dict[str, np.ndarray],
    meta: dict[str, dict[str, Any]],
    k: int = 5,
) -> list[dict[str, Any]]:
    source = vectors.get(store_id)
    if source is None or source.size == 0:
        return []
    scored: list[tuple[str, float]] = []
    for other_id, other in vectors.items():
        if other_id == store_id:
            continue
        scored.append((other_id, _cosine(source, other)))
    scored.sort(key=lambda item: item[1], reverse=True)
    peers: list[dict[str, Any]] = []
    for other_id, similarity in scored[:k]:
        info = meta.get(other_id, {})
        peers.append(
            {
                "store_id": other_id,
                "store_name": info.get("store_name", other_id),
                "retailer": info.get("retailer", ""),
                "city": info.get("city", ""),
                "state": info.get("state", ""),
                "similarity": round(similarity, 3),
            }
        )
    return peers


def _build_cache() -> ApiCache:
    kpi_table = _load_curated_kpi_table()
    if kpi_table.empty:
        raise RuntimeError(
            "No curated store KPI data found under data/USA_100_Stores/curated. "
            "Run Stage 2 curation first."
        )

    review_insights_by_store = _build_store_review_insights(_load_reviews_frame())
    news_insights_by_store = build_news_insights(_load_news_frame())

    embedding_docs, vectors = _load_store_embeddings()
    if not embedding_docs:
        _load_env()
        raise RuntimeError(
            "No documents in store embedding index "
            f"'{os.getenv('AZURE_SEARCH_STORE_INDEX_NAME', 'store-dna-store-vectors')}'. "
            "Run Stage 6 Vector Index notebook first."
        )

    # Meta from embeddings (preferred) with curated fallback
    meta: dict[str, dict[str, Any]] = {}
    for doc in embedding_docs:
        store_id = str(doc["store_id"])
        meta[store_id] = {
            "store_name": str(doc.get("store_name", store_id)),
            "retailer": str(doc.get("retailer", "")),
            "city": str(doc.get("city", "")),
            "state": str(doc.get("state", "")),
            "format": str(doc.get("store_format", "")),
        }
    for _, row in kpi_table.iterrows():
        store_id = str(row["store_id"])
        if store_id not in meta:
            meta[store_id] = {
                "store_name": str(row.get("store_name", store_id)),
                "retailer": str(row.get("retailer", "")),
                "city": str(row.get("city", "")),
                "state": str(row.get("state", "")),
                "format": str(row.get("store_format", "")),
            }

    kpi_lookup = kpi_table.set_index("store_id").to_dict(orient="index")
    network_keys = [
        "fulfillment_rate",
        "oos_rate",
        "shrink_pct",
        "positive_review_pct",
        "complaints_4w",
        "review_count",
        "positive",
        "negative",
        "negative_review_pct",
        "news_count",
        "local_news_count",
        "competitor_mention_count",
        "negative_news_count",
    ]
    network: dict[str, float] = {}
    for key in network_keys:
        if key in kpi_table.columns:
            network[key] = float(kpi_table[key].mean())
        else:
            network[key] = 0.0

    # Prefer store ids that exist in either source; keep curated order when possible
    store_ids = [str(x) for x in kpi_table["store_id"].tolist()]
    for store_id in vectors:
        if store_id not in store_ids:
            store_ids.append(store_id)

    stores: list[dict[str, Any]] = []
    status_rollups = {"Doing well": 0, "Steady": 0, "Needs focus": 0}
    retailers: dict[str, int] = defaultdict(int)

    for store_id in store_ids:
        info = meta.get(store_id, {})
        row = kpi_lookup.get(store_id, {})
        kpis = {key: float(row.get(key, 0.0)) for key in network_keys}
        # pass through raw sentiment counts for pulse
        for key in ("positive", "neutral", "negative"):
            kpis[key] = float(row.get(key, 0.0))

        executive = _build_executive(kpis, network)
        status_rollups[executive["status"]] = status_rollups.get(executive["status"], 0) + 1
        retailers[str(info.get("retailer", ""))] += 1
        peers = _top_peers(store_id, vectors, meta, k=5)
        similar_to = {peer["store_id"]: peer["similarity"] for peer in peers}

        stores.append(
            {
                "store_id": store_id,
                "store_name": str(info.get("store_name", store_id)),
                "retailer": str(info.get("retailer", "")),
                "city": str(info.get("city", "")),
                "state": str(info.get("state", "")),
                "format": str(info.get("format", "")),
                "top_peers": peers,
                "similar_to": similar_to,
                "executive": executive,
                "business": {
                    "store_pulse": _build_pulse(
                        kpis,
                        network,
                        review_insights_by_store.get(store_id),
                        news_insights_by_store.get(store_id),
                    ),
                    "focus_areas": _focus_areas(executive),
                },
            }
        )

    stores.sort(key=lambda s: s["store_id"])
    _load_env()
    summary = {
        "store_count": len(stores),
        "retailer_count": len([name for name in retailers if name]),
        "retailers": [{"name": name, "count": count} for name, count in sorted(retailers.items()) if name],
        "status_rollups": status_rollups,
        "reporting_window": "Past 4 weeks (curated data)",
        "search_index": os.getenv("AZURE_SEARCH_STORE_INDEX_NAME", "store-dna-store-vectors"),
        "kpi_source": "curated_csv",
        "similarity_source": "azure_store_embeddings",
    }
    return ApiCache(loaded_at=time.time(), summary=summary, stores=stores, vectors=vectors)


def get_cache(force_refresh: bool = False) -> ApiCache:
    global _CACHE
    if _CACHE is None or force_refresh or (time.time() - _CACHE.loaded_at) > _CACHE_TTL_SEC:
        _CACHE = _build_cache()
    return _CACHE


def get_health() -> dict[str, Any]:
    cache = get_cache()
    return {
        "status": "ok",
        "store_count": cache.summary["store_count"],
        "search_index": cache.summary["search_index"],
        "kpi_source": "curated_csv",
        "similarity_source": "azure_store_embeddings",
        "cached_at": cache.loaded_at,
    }


def get_portfolio_payload() -> dict[str, Any]:
    cache = get_cache()
    return {"summary": cache.summary, "stores": cache.stores}


def get_store(store_id: str) -> dict[str, Any] | None:
    for store in get_cache().stores:
        if store["store_id"] == store_id:
            return store
    return None


def compare_stores(store_a: str, store_b: str, *, include_learnings: bool = True) -> dict[str, Any]:
    cache = get_cache()
    vec_a = cache.vectors.get(store_a)
    vec_b = cache.vectors.get(store_b)
    similarity = _cosine(vec_a, vec_b) if vec_a is not None and vec_b is not None else None
    doc_a = get_store(store_a)
    doc_b = get_store(store_b)
    learnings = None
    if include_learnings and doc_a is not None and doc_b is not None:
        learnings = build_peer_learnings(doc_a, doc_b, similarity)
    return {
        "store_a": store_a,
        "store_b": store_b,
        "similarity": round(similarity, 3) if similarity is not None else None,
        "store_a_doc": doc_a,
        "store_b_doc": doc_b,
        "peer_learnings": learnings,
    }
