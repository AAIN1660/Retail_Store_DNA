"""
Hybrid Store DNA vectors — readable KPI head + semantic modality tail.

Builds one hybrid vector per store:
  [HEAD scalars | projected TAIL from all 6 Stage 4 modality vectors]

Uploads to a *separate* Azure AI Search index (default: store-dna-hybrid-vectors)
so the original Stage 6 index (store-dna-store-vectors) is left unchanged.
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
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from sklearn.random_projection import GaussianRandomProjection

from src.store_dna_builder import ALL_MODALITIES, EMBEDDING_MODALITIES, STRUCTURED_MODALITY, l2_normalize
from src.store_dna_search import sanitize_search_document_id

# ---------------------------------------------------------------------------
# HEAD slot schema — fixed order so KPIs can be extracted by index
# ---------------------------------------------------------------------------

HEAD_SLOT_SCHEMA: tuple[dict[str, str], ...] = (
    {"name": "fulfillment_rate", "modality": "structured_ops", "description": "4-week avg fulfillment rate"},
    {"name": "oos_rate", "modality": "structured_ops", "description": "4-week avg out-of-stock rate"},
    {"name": "shrink_pct", "modality": "structured_ops", "description": "4-week avg shrink %"},
    {"name": "labor_hours_4w", "modality": "structured_ops", "description": "4-week avg labor hours"},
    {"name": "complaints_4w", "modality": "structured_ops", "description": "4-week sum of complaints"},
    {"name": "positive_review_pct", "modality": "reviews", "description": "Share of positive reviews"},
    {"name": "negative_review_pct", "modality": "reviews", "description": "Share of negative reviews"},
    {"name": "review_count", "modality": "reviews", "description": "Total review documents"},
    {"name": "news_count", "modality": "news", "description": "Total news documents"},
    {"name": "local_news_count", "modality": "news", "description": "Local news story count"},
    {"name": "competitor_mention_count", "modality": "news", "description": "Competitor mention count"},
    {"name": "negative_news_count", "modality": "news", "description": "Negative-impact news count"},
    {"name": "product_count", "modality": "products", "description": "Product / SKU row count"},
    {"name": "product_in_stock_pct", "modality": "products", "description": "Share of products in stock"},
    {"name": "report_count", "modality": "reports", "description": "Internal report count"},
    {"name": "high_severity_report_count", "modality": "reports", "description": "High-severity report count"},
    {"name": "ops_weeks_count", "modality": "ops_weekly", "description": "Ops weeks available"},
    {"name": "reviews_present", "modality": "reviews", "description": "1 if reviews modality vector non-zero"},
    {"name": "news_present", "modality": "news", "description": "1 if news modality vector non-zero"},
    {"name": "reports_present", "modality": "reports", "description": "1 if reports modality vector non-zero"},
    {"name": "products_present", "modality": "products", "description": "1 if products modality vector non-zero"},
    {"name": "ops_weekly_present", "modality": "ops_weekly", "description": "1 if ops_weekly modality vector non-zero"},
    {"name": "structured_ops_present", "modality": "structured_ops", "description": "1 if structured_ops vector non-zero"},
)

HEAD_DIM = len(HEAD_SLOT_SCHEMA)
DEFAULT_HYBRID_INDEX_NAME = "store-dna-hybrid-vectors"
DEFAULT_TOTAL_VECTOR_DIM = 2048


@dataclass
class HybridVectorConfig:
    """Hybrid vector + Azure index settings (separate from Stage 6)."""

    search_endpoint: str
    search_api_key: str
    index_name: str = DEFAULT_HYBRID_INDEX_NAME
    total_vector_dimensions: int = DEFAULT_TOTAL_VECTOR_DIM
    upload_batch_size: int = 200
    projection_random_state: int = 42
    include_structured_ops_in_tail: bool = True

    @classmethod
    def from_env(cls, project_root: Path | None = None, **overrides: Any) -> HybridVectorConfig:
        if project_root:
            load_dotenv(project_root / ".env")
        index_name = overrides.pop(
            "index_name",
            os.getenv("AZURE_SEARCH_HYBRID_INDEX_NAME", DEFAULT_HYBRID_INDEX_NAME),
        )
        total_vector_dimensions = overrides.pop(
            "total_vector_dimensions",
            int(os.getenv("AZURE_SEARCH_HYBRID_VECTOR_DIMENSIONS", str(DEFAULT_TOTAL_VECTOR_DIM))),
        )
        upload_batch_size = overrides.pop(
            "upload_batch_size",
            int(os.getenv("AZURE_SEARCH_UPLOAD_BATCH_SIZE", "500")),
        )
        return cls(
            search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/"),
            search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
            index_name=index_name,
            total_vector_dimensions=total_vector_dimensions,
            upload_batch_size=upload_batch_size,
            **overrides,
        )

    @property
    def head_dim(self) -> int:
        return HEAD_DIM

    @property
    def tail_dim(self) -> int:
        return int(self.total_vector_dimensions) - HEAD_DIM


def head_slot_index(name: str) -> int:
    """Return HEAD slot index for a named KPI field."""
    for i, slot in enumerate(HEAD_SLOT_SCHEMA):
        if slot["name"] == name:
            return i
    raise KeyError(f"Unknown HEAD slot: {name}")


def extract_kpis_from_hybrid_vector(vector: list[float] | np.ndarray) -> dict[str, float]:
    """Extract readable KPIs from the HEAD portion of a hybrid vector."""
    arr = np.asarray(vector, dtype=np.float64)
    if arr.size < HEAD_DIM:
        raise ValueError(f"Vector too short for HEAD ({arr.size} < {HEAD_DIM})")
    return {slot["name"]: float(arr[i]) for i, slot in enumerate(HEAD_SLOT_SCHEMA)}


def _recent_ops_metrics(ops: pd.DataFrame) -> pd.DataFrame:
    ops = ops.copy()
    ops["week_end_date"] = pd.to_datetime(ops["week_end_date"])
    latest = ops.groupby("store_id")["week_end_date"].transform("max")
    recent = ops[ops["week_end_date"] >= latest - pd.Timedelta(weeks=3)]
    metrics = recent.groupby("store_id").agg(
        fulfillment_rate=("fulfillment_rate", "mean"),
        oos_rate=("oos_rate", "mean"),
        shrink_pct=("shrink_pct", "mean"),
        labor_hours_4w=("labor_hours", "mean"),
        complaints_4w=("customer_complaints", "sum"),
    )
    weeks = ops.groupby("store_id").size().rename("ops_weeks_count")
    return metrics.join(weeks, how="left").fillna(0.0)


def _review_metrics(reviews: pd.DataFrame) -> pd.DataFrame:
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
    return counts[["review_count", "positive_review_pct", "negative_review_pct"]]


def _news_metrics(news: pd.DataFrame) -> pd.DataFrame:
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


def _product_metrics(products: pd.DataFrame) -> pd.DataFrame:
    if "in_stock" in products.columns:
        in_stock = products.groupby("store_id")["in_stock"].apply(
            lambda s: float(pd.Series(s).astype(str).str.lower().isin(["true", "1", "yes"]).mean())
        )
    else:
        in_stock = products.groupby("store_id").size() * 0.0
    counts = products.groupby("store_id").size().rename("product_count")
    return pd.DataFrame({"product_count": counts, "product_in_stock_pct": in_stock}).fillna(0.0)


def _report_metrics(reports: pd.DataFrame) -> pd.DataFrame:
    counts = reports.groupby("store_id").size().rename("report_count")
    if "severity" in reports.columns:
        high = (
            reports.assign(_high=reports["severity"].astype(str).str.lower().eq("high"))
            .groupby("store_id")["_high"]
            .sum()
            .rename("high_severity_report_count")
        )
    else:
        high = counts * 0
    return pd.DataFrame({"report_count": counts, "high_severity_report_count": high}).fillna(0.0)


def build_head_matrix(
    store_ids: list[str],
    curated_dir: Path,
    modality_present: dict[str, np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build HEAD KPI matrix (n_stores × HEAD_DIM) from curated tables."""
    ops = pd.read_csv(curated_dir / "fact_operations_weekly.csv")
    reviews_path = curated_dir / "stg_reviews.csv"
    news_path = curated_dir / "stg_news.csv"
    products_path = curated_dir / "stg_products.csv"
    reports_path = curated_dir / "stg_reports.csv"

    # Prefer enriched sentiment when available
    enriched_reviews = curated_dir.parent / "enriched" / "reviews_enriched.csv"
    enriched_news = curated_dir.parent / "enriched" / "news_enriched.csv"
    if enriched_reviews.exists():
        reviews = pd.read_csv(enriched_reviews)
    else:
        reviews = pd.read_csv(reviews_path)
    if enriched_news.exists():
        news = pd.read_csv(enriched_news)
    else:
        news = pd.read_csv(news_path)

    products = pd.read_csv(products_path)
    reports = pd.read_csv(reports_path)

    ops_m = _recent_ops_metrics(ops)
    rev_m = _review_metrics(reviews)
    news_m = _news_metrics(news)
    prod_m = _product_metrics(products)
    rep_m = _report_metrics(reports)

    frame = pd.DataFrame({"store_id": store_ids}).set_index("store_id")
    for part in (ops_m, rev_m, news_m, prod_m, rep_m):
        frame = frame.join(part, how="left")
    frame = frame.fillna(0.0)

    for modality, flags in modality_present.items():
        frame[f"{modality}_present"] = flags.astype(float)

    rows: list[list[float]] = []
    for sid in store_ids:
        row = frame.loc[sid] if sid in frame.index else None
        values: list[float] = []
        for slot in HEAD_SLOT_SCHEMA:
            name = slot["name"]
            if row is None or name not in frame.columns:
                values.append(0.0)
            else:
                values.append(float(row[name]))
        rows.append(values)

    head = np.asarray(rows, dtype=np.float32)
    return head, frame.reset_index()


def load_stage4_modality_matrices(
    modality_dir: Path,
    store_ids: list[str],
) -> dict[str, np.ndarray]:
    """Load and align Stage 4 modality vectors to store_ids order."""
    aligned: dict[str, np.ndarray] = {}
    for modality in ALL_MODALITIES:
        path = modality_dir / f"{modality}_vectors.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 4 output: {path}")
        data = np.load(path)
        modality_ids = [str(x) for x in data["store_ids"]]
        vectors = data["vectors"].astype(np.float32)
        index = {sid: i for i, sid in enumerate(modality_ids)}
        matrix = np.zeros((len(store_ids), vectors.shape[1]), dtype=np.float32)
        for row_idx, sid in enumerate(store_ids):
            if sid in index:
                matrix[row_idx] = vectors[index[sid]]
        aligned[modality] = matrix
    return aligned


def build_semantic_tail(
    aligned: dict[str, np.ndarray],
    target_tail_dim: int,
    *,
    include_structured_ops: bool = True,
    projection_random_state: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Concatenate all modality vectors, then project to target_tail_dim.

    HEAD is NOT included here — only the semantic / modality fingerprint.
    """
    blocks: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    modalities = list(EMBEDDING_MODALITIES)
    if include_structured_ops:
        modalities = modalities + [STRUCTURED_MODALITY]

    for modality in modalities:
        matrix = l2_normalize(aligned[modality])
        blocks.append(matrix)
        meta.append({"modality": modality, "vector_dim": int(matrix.shape[1])})

    raw = np.concatenate(blocks, axis=1).astype(np.float32)
    source_dim = int(raw.shape[1])

    if source_dim <= target_tail_dim:
        # Pad with zeros if somehow smaller than target
        padded = np.zeros((raw.shape[0], target_tail_dim), dtype=np.float32)
        padded[:, :source_dim] = raw
        projected = l2_normalize(padded)
        method = "identity_pad"
    else:
        projector = GaussianRandomProjection(
            n_components=target_tail_dim,
            random_state=projection_random_state,
        )
        projected = projector.fit_transform(raw).astype(np.float32)
        projected = l2_normalize(projected)
        method = "gaussian_random_projection"

    return projected, {
        "method": method,
        "source_dim": source_dim,
        "target_dim": target_tail_dim,
        "modalities": meta,
        "include_structured_ops": include_structured_ops,
    }


def assemble_hybrid_vectors(head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Concatenate HEAD + TAIL into one hybrid vector per store."""
    if head.shape[0] != tail.shape[0]:
        raise ValueError("HEAD and TAIL store counts differ")
    return np.concatenate([head.astype(np.float32), tail.astype(np.float32)], axis=1)


def ensure_hybrid_vector_index(config: HybridVectorConfig) -> None:
    """Create or update the hybrid store vector index (separate from Stage 6)."""
    index_client = SearchIndexClient(
        endpoint=config.search_endpoint,
        credential=AzureKeyCredential(config.search_api_key),
    )

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-hybrid")],
        profiles=[
            VectorSearchProfile(
                name="store-dna-hybrid-profile",
                algorithm_configuration_name="hnsw-hybrid",
            )
        ],
    )

    # Named KPI fields mirror HEAD slots for easy UI retrieval
    kpi_fields = [
        SimpleField(name=slot["name"], type=SearchFieldDataType.Double, filterable=True, retrievable=True)
        for slot in HEAD_SLOT_SCHEMA
    ]

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="store_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="retailer", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="banner", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="store_name", type=SearchFieldDataType.String),
        SimpleField(name="city", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="state", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="store_format", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="fusion_method", type=SearchFieldDataType.String),
        SimpleField(name="head_dim", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="tail_dim", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="hybrid_dim", type=SearchFieldDataType.Int32, filterable=True),
        *kpi_fields,
        SearchField(
            name="store_dna_hybrid_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=True,
            vector_search_dimensions=config.total_vector_dimensions,
            vector_search_profile_name="store-dna-hybrid-profile",
        ),
    ]

    index = SearchIndex(
        name=config.index_name,
        fields=fields,
        vector_search=vector_search,
    )
    index_client.create_or_update_index(index)


def build_hybrid_documents(
    store_ids: list[str],
    hybrid_vectors: np.ndarray,
    head: np.ndarray,
    dim_store: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build Azure documents with named KPIs + hybrid vector."""
    dim_lookup = dim_store.set_index("store_id").to_dict(orient="index")
    documents: list[dict[str, Any]] = []
    for i, store_id in enumerate(store_ids):
        dim_row = dim_lookup.get(store_id, {})
        doc: dict[str, Any] = {
            "id": sanitize_search_document_id(f"store_dna_hybrid_{store_id}"),
            "store_id": store_id,
            "retailer": str(dim_row.get("retailer", "")),
            "banner": str(dim_row.get("banner", "")),
            "store_name": str(dim_row.get("store_name", "")),
            "city": str(dim_row.get("city", "")),
            "state": str(dim_row.get("state", "")),
            "store_format": str(dim_row.get("store_format", "")),
            "fusion_method": "hybrid_head_plus_projected_modality_tail",
            "head_dim": HEAD_DIM,
            "tail_dim": int(hybrid_vectors.shape[1] - HEAD_DIM),
            "hybrid_dim": int(hybrid_vectors.shape[1]),
            "store_dna_hybrid_vector": hybrid_vectors[i].astype(float).tolist(),
        }
        for j, slot in enumerate(HEAD_SLOT_SCHEMA):
            doc[slot["name"]] = float(head[i, j])
        documents.append(doc)
    return documents


def upload_hybrid_documents(
    config: HybridVectorConfig,
    documents: list[dict[str, Any]],
    pause_sec: float = 0.2,
) -> int:
    """Upload hybrid documents in batches."""
    if not documents:
        return 0
    client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.search_api_key),
    )
    uploaded = 0
    for start in range(0, len(documents), config.upload_batch_size):
        batch = documents[start : start + config.upload_batch_size]
        client.merge_or_upload_documents(batch)
        uploaded += len(batch)
        if start + config.upload_batch_size < len(documents):
            time.sleep(pause_sec)
    return uploaded


def get_hybrid_index_count(config: HybridVectorConfig) -> int:
    client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.search_api_key),
    )
    return client.get_document_count()


def run_hybrid_vector_pipeline(
    curated_dir: Path,
    modality_dir: Path,
    output_dir: Path,
    config: HybridVectorConfig,
) -> dict[str, Any]:
    """
    Full hybrid pipeline:
      1) Load Stage 4 modality vectors (all 6)
      2) Build extractable HEAD from curated KPIs / modality coverage
      3) Build projected TAIL from all modality vectors
      4) Assemble hybrid vectors
      5) Create separate Azure index and upload
    """
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    dim_store = pd.read_csv(curated_dir / "dim_store.csv")
    store_ids = [str(x) for x in dim_store["store_id"].tolist()]

    print(f"[Hybrid] Loading Stage 4 modality vectors for {len(store_ids)} stores...")
    aligned = load_stage4_modality_matrices(modality_dir, store_ids)
    modality_present = {
        modality: (np.linalg.norm(aligned[modality], axis=1) > 0).astype(np.float32)
        for modality in ALL_MODALITIES
    }

    print(f"[Hybrid] Building HEAD ({HEAD_DIM} KPI / coverage slots)...")
    head, head_frame = build_head_matrix(store_ids, curated_dir, modality_present)

    print(f"[Hybrid] Building TAIL (all modality vectors → {config.tail_dim} dims)...")
    tail, tail_meta = build_semantic_tail(
        aligned,
        config.tail_dim,
        include_structured_ops=config.include_structured_ops_in_tail,
        projection_random_state=config.projection_random_state,
    )

    hybrid = assemble_hybrid_vectors(head, tail)
    print(f"[Hybrid] Assembled hybrid shape: {hybrid.shape} (head={HEAD_DIM}, tail={config.tail_dim})")

    np.savez_compressed(
        output_dir / "store_dna_hybrid_vectors.npz",
        store_ids=np.array(store_ids),
        vectors=hybrid,
        head=head,
        tail=tail,
    )
    head_frame.to_csv(output_dir / "hybrid_head_kpis.csv", index=False)
    schema_path = output_dir / "hybrid_head_slot_schema.json"
    schema_path.write_text(json.dumps(list(HEAD_SLOT_SCHEMA), indent=2), encoding="utf-8")

    print(f"[Hybrid] Ensuring Azure index: {config.index_name}")
    ensure_hybrid_vector_index(config)
    documents = build_hybrid_documents(store_ids, hybrid, head, dim_store)
    print(f"[Hybrid] Uploading {len(documents)} documents...")
    uploaded = upload_hybrid_documents(config, documents)
    count = get_hybrid_index_count(config)

    manifest = {
        "index_name": config.index_name,
        "fusion_method": "hybrid_head_plus_projected_modality_tail",
        "head_dim": HEAD_DIM,
        "tail_dim": config.tail_dim,
        "hybrid_dim": int(hybrid.shape[1]),
        "head_slot_schema": list(HEAD_SLOT_SCHEMA),
        "tail_projection": tail_meta,
        "stores": len(store_ids),
        "documents_uploaded": uploaded,
        "index_document_count": count,
        "outputs": {
            "vectors": "store_dna_hybrid_vectors.npz",
            "head_kpis": "hybrid_head_kpis.csv",
            "schema": "hybrid_head_slot_schema.json",
        },
        "note": (
            "Separate from Stage 6 index store-dna-store-vectors. "
            "Extract KPIs from vector[0:head_dim] or named Azure fields."
        ),
        "elapsed_sec": round(time.time() - start, 1),
    }
    (output_dir / "hybrid_vector_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"[Hybrid] Complete in {manifest['elapsed_sec']}s → index={config.index_name}")
    return manifest
