"""
Stage 4 — Modality Vectors for Retail StoreDNA.

Mean-pools row-level embeddings from Azure AI Search into one vector per
store × modality, and builds a normalized structured ops vector from
fact_operations_weekly KPIs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.store_dna_search import SearchIndexConfig, fetch_documents_by_modality

EMBEDDING_MODALITIES = ("reviews", "news", "reports", "products", "ops_weekly")
STRUCTURED_MODALITY = "structured_ops"
OPS_KPI_COLUMNS = (
    "labor_hours",
    "shrink_pct",
    "oos_rate",
    "fulfillment_rate",
    "customer_complaints",
)


@dataclass
class ModalityVectorConfig:
    """Stage 4 settings."""

    search_endpoint: str
    search_api_key: str
    search_index_name: str
    embedding_dimensions: int = 3072
    max_stores: int | None = None
    l2_normalize_pooled: bool = True
    pool_method: str = "mean"

    @classmethod
    def from_env(cls, project_root: Path | None = None, **overrides: Any) -> ModalityVectorConfig:
        import os

        if project_root:
            load_dotenv(project_root / ".env")
        return cls(
            search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/"),
            search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
            search_index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "store-dna-embeddings"),
            embedding_dimensions=int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "3072")),
            **overrides,
        )

    def search_config(self) -> SearchIndexConfig:
        return SearchIndexConfig(
            endpoint=self.search_endpoint,
            api_key=self.search_api_key,
            index_name=self.search_index_name,
            vector_dimensions=self.embedding_dimensions,
        )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def mean_pool_by_store(documents: list[dict]) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Group row vectors by store_id and mean-pool."""
    buckets: dict[str, list[np.ndarray]] = {}
    for doc in documents:
        store_id = str(doc.get("store_id", ""))
        vector = doc.get("content_vector")
        if not store_id or vector is None:
            continue
        buckets.setdefault(store_id, []).append(np.asarray(vector, dtype=np.float32))

    store_ids = sorted(buckets)
    if not store_ids:
        return [], np.empty((0, 0), dtype=np.float32), {}

    vectors = np.stack([np.mean(buckets[sid], axis=0) for sid in store_ids]).astype(np.float32)
    doc_counts = {sid: len(buckets[sid]) for sid in store_ids}
    return store_ids, vectors, doc_counts


def build_structured_ops_vectors(
    ops_df: pd.DataFrame,
    store_ids: list[str],
) -> np.ndarray:
    """Aggregate weekly KPIs per store and z-score normalize across stores."""
    agg = ops_df.groupby("store_id", as_index=True)[list(OPS_KPI_COLUMNS)].mean()
    agg = agg.reindex(store_ids)
    filled = agg.fillna(agg.mean())
    mean = filled.mean()
    std = filled.std(ddof=0).replace(0, 1.0)
    normalized = ((filled - mean) / std).fillna(0.0)
    return normalized.to_numpy(dtype=np.float32)


def pool_modality_from_search(
    config: ModalityVectorConfig,
    modality: str,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Fetch row vectors from Azure AI Search and mean-pool per store."""
    documents = fetch_documents_by_modality(config.search_config(), modality)
    store_ids, vectors, doc_counts = mean_pool_by_store(documents)
    if config.max_stores and store_ids:
        keep = store_ids[: config.max_stores]
        idx = [store_ids.index(sid) for sid in keep]
        store_ids = keep
        vectors = vectors[idx]
        doc_counts = {sid: doc_counts[sid] for sid in keep}
    if config.l2_normalize_pooled and len(vectors):
        vectors = l2_normalize(vectors)
    return store_ids, vectors, doc_counts


def save_modality_npz(path: Path, store_ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, store_ids=np.array(store_ids), vectors=vectors)


def run_modality_vectors(
    curated_dir: Path,
    output_dir: Path,
    config: ModalityVectorConfig,
) -> dict[str, Any]:
    """Full Stage 4 pipeline."""
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    dim_store = pd.read_csv(curated_dir / "dim_store.csv")
    ops_df = pd.read_csv(curated_dir / "fact_operations_weekly.csv", parse_dates=["week_end_date"])

    if config.max_stores:
        store_ids = dim_store["store_id"].head(config.max_stores).tolist()
        ops_df = ops_df[ops_df["store_id"].isin(store_ids)]
        print(f"[Stage 4] Store filter applied: {len(store_ids)} stores")
    else:
        store_ids = dim_store["store_id"].tolist()
        print(f"[Stage 4] Processing {len(store_ids)} stores")

    index_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "search_index": config.search_index_name,
        "pool_method": config.pool_method,
        "l2_normalize_pooled": config.l2_normalize_pooled,
        "modalities": {},
    }

    for modality in EMBEDDING_MODALITIES:
        modality_start = time.time()
        print(f"[Stage 4] {modality}: fetching row vectors from Azure AI Search...")
        pooled_ids, vectors, doc_counts = pool_modality_from_search(config, modality)
        if not pooled_ids:
            print(f"[Stage 4] {modality}: no vectors found — skipping")
            continue

        # Align to dim_store order where possible
        id_to_row = {sid: i for i, sid in enumerate(pooled_ids)}
        aligned = np.zeros((len(store_ids), vectors.shape[1]), dtype=np.float32)
        present = 0
        for i, sid in enumerate(store_ids):
            if sid in id_to_row:
                aligned[i] = vectors[id_to_row[sid]]
                present += 1

        out_path = output_dir / f"{modality}_vectors.npz"
        save_modality_npz(out_path, store_ids, aligned)
        elapsed = time.time() - modality_start
        print(
            f"[Stage 4] {modality}: pooled {sum(doc_counts.values()):,} rows -> "
            f"{present}/{len(store_ids)} stores, dim={aligned.shape[1]} in {elapsed:.1f}s"
        )

        for sid in store_ids:
            index_rows.append({
                "store_id": sid,
                "modality": modality,
                "doc_count": doc_counts.get(sid, 0),
                "vector_dim": int(aligned.shape[1]),
                "has_vector": sid in id_to_row,
            })

        manifest["modalities"][modality] = {
            "rows_pooled": int(sum(doc_counts.values())),
            "stores_with_vector": present,
            "vector_dim": int(aligned.shape[1]),
            "file": out_path.name,
        }

    print("[Stage 4] structured_ops: building KPI vector from fact_operations_weekly...")
    structured = build_structured_ops_vectors(ops_df, store_ids)
    structured_path = output_dir / f"{STRUCTURED_MODALITY}_vectors.npz"
    save_modality_npz(structured_path, store_ids, structured)

    for sid in store_ids:
        weeks = int((ops_df["store_id"] == sid).sum())
        index_rows.append({
            "store_id": sid,
            "modality": STRUCTURED_MODALITY,
            "doc_count": weeks,
            "vector_dim": int(structured.shape[1]),
            "has_vector": weeks > 0,
        })

    manifest["modalities"][STRUCTURED_MODALITY] = {
        "kpi_columns": list(OPS_KPI_COLUMNS),
        "stores_with_vector": int((ops_df.groupby("store_id").size().reindex(store_ids, fill_value=0) > 0).sum()),
        "vector_dim": int(structured.shape[1]),
        "file": structured_path.name,
    }

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(output_dir / "store_modality_index.csv", index=False)

    manifest["store_count"] = len(store_ids)
    manifest["elapsed_sec"] = round(time.time() - start_time, 1)
    (output_dir / "modality_vectors_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"[Stage 4] Wrote outputs to {output_dir}")
    print(f"[Stage 4] Complete in {manifest['elapsed_sec']}s")
    return manifest
