"""
Stage 6 - Vector Index for Retail StoreDNA.

Creates a store-level Azure AI Search index for Stage 5 fused StoreDNA vectors
and uploads one document per store.
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

from src.store_dna_search import sanitize_search_document_id


@dataclass
class StoreVectorIndexConfig:
    """Stage 6 settings."""

    search_endpoint: str
    search_api_key: str
    index_name: str = "store-dna-store-vectors"
    vector_dimensions: int = 2048
    upload_batch_size: int = 200
    projection_random_state: int = 42

    @classmethod
    def from_env(cls, project_root: Path | None = None, **overrides: Any) -> StoreVectorIndexConfig:
        if project_root:
            load_dotenv(project_root / ".env")
        index_name = overrides.pop("index_name", os.getenv("AZURE_SEARCH_STORE_INDEX_NAME", "store-dna-store-vectors"))
        upload_batch_size = overrides.pop("upload_batch_size", int(os.getenv("AZURE_SEARCH_UPLOAD_BATCH_SIZE", "500")))
        vector_dimensions = overrides.pop(
            "vector_dimensions",
            int(os.getenv("AZURE_SEARCH_STORE_VECTOR_DIMENSIONS", "2048")),
        )
        projection_random_state = overrides.pop("projection_random_state", 42)
        return cls(
            search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/"),
            search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
            index_name=index_name,
            vector_dimensions=vector_dimensions,
            upload_batch_size=upload_batch_size,
            projection_random_state=projection_random_state,
            **overrides,
        )


def ensure_store_vector_index(config: StoreVectorIndexConfig) -> None:
    """Create or update the Stage 6 store-level vector index."""
    index_client = SearchIndexClient(
        endpoint=config.search_endpoint,
        credential=AzureKeyCredential(config.search_api_key),
    )

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-default")],
        profiles=[
            VectorSearchProfile(
                name="store-dna-profile",
                algorithm_configuration_name="hnsw-default",
            )
        ],
    )

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
        SimpleField(name="store_dna_dim", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="store_dna_norm", type=SearchFieldDataType.Double, filterable=True),
        SearchField(
            name="store_dna_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=True,
            vector_search_dimensions=config.vector_dimensions,
            vector_search_profile_name="store-dna-profile",
        ),
    ]

    index = SearchIndex(
        name=config.index_name,
        fields=fields,
        vector_search=vector_search,
    )
    index_client.create_or_update_index(index)


def load_store_dna_outputs(store_dna_dir: Path) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Load Stage 5 outputs."""
    npz = np.load(store_dna_dir / "store_dna_vectors.npz")
    store_ids = [str(x) for x in npz["store_ids"]]
    vectors = npz["vectors"].astype(np.float32)
    index_df = pd.read_csv(store_dna_dir / "store_dna_index.csv")
    return store_ids, vectors, index_df


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def project_vectors_for_search(vectors: np.ndarray, config: StoreVectorIndexConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Project vectors to an Azure AI Search compatible dimension if needed."""
    source_dim = int(vectors.shape[1])
    target_dim = int(config.vector_dimensions)
    if source_dim <= target_dim:
        return vectors.astype(np.float32), {
            "projection_applied": False,
            "source_dim": source_dim,
            "target_dim": source_dim,
            "method": "identity",
        }

    projector = GaussianRandomProjection(
        n_components=target_dim,
        random_state=config.projection_random_state,
    )
    projected = projector.fit_transform(vectors).astype(np.float32)
    projected = l2_normalize(projected)
    return projected, {
        "projection_applied": True,
        "source_dim": source_dim,
        "target_dim": target_dim,
        "method": "gaussian_random_projection",
        "random_state": config.projection_random_state,
    }


def build_store_vector_documents(
    store_ids: list[str],
    vectors: np.ndarray,
    dim_store: pd.DataFrame,
    store_dna_index: pd.DataFrame,
    fusion_method: str,
) -> list[dict[str, Any]]:
    """Build store-level Azure AI Search documents."""
    dim_lookup = dim_store.set_index("store_id").to_dict(orient="index")
    index_lookup = store_dna_index.set_index("store_id").to_dict(orient="index")

    documents: list[dict[str, Any]] = []
    for i, store_id in enumerate(store_ids):
        dim_row = dim_lookup.get(store_id, {})
        qc_row = index_lookup.get(store_id, {})
        documents.append(
            {
                "id": sanitize_search_document_id(f"store_dna_{store_id}"),
                "store_id": store_id,
                "retailer": str(dim_row.get("retailer", "")),
                "banner": str(dim_row.get("banner", "")),
                "store_name": str(dim_row.get("store_name", "")),
                "city": str(dim_row.get("city", "")),
                "state": str(dim_row.get("state", "")),
                "store_format": str(dim_row.get("store_format", "")),
                "fusion_method": fusion_method,
                "store_dna_dim": int(vectors.shape[1]),
                "store_dna_norm": float(qc_row.get("store_dna_norm", float(np.linalg.norm(vectors[i])))),
                "store_dna_vector": vectors[i].astype(float).tolist(),
            }
        )
    return documents


def upload_store_vector_documents(
    config: StoreVectorIndexConfig,
    documents: list[dict[str, Any]],
    pause_sec: float = 0.2,
) -> int:
    """Upload store-level vector documents in batches."""
    if not documents:
        return 0

    search_client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.search_api_key),
    )

    uploaded = 0
    for start in range(0, len(documents), config.upload_batch_size):
        batch = documents[start : start + config.upload_batch_size]
        search_client.merge_or_upload_documents(batch)
        uploaded += len(batch)
        if start + config.upload_batch_size < len(documents):
            time.sleep(pause_sec)
    return uploaded


def get_store_vector_index_count(config: StoreVectorIndexConfig) -> int:
    """Return document count in the Stage 6 index."""
    search_client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.search_api_key),
    )
    return search_client.get_document_count()


def run_store_vector_index(
    curated_dir: Path,
    store_dna_dir: Path,
    output_dir: Path,
    config: StoreVectorIndexConfig,
) -> dict[str, Any]:
    """Full Stage 6 pipeline."""
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    store_ids, vectors, store_dna_index = load_store_dna_outputs(store_dna_dir)
    dim_store = pd.read_csv(curated_dir / "dim_store.csv")
    manifest = json.loads((store_dna_dir / "store_dna_manifest.json").read_text(encoding="utf-8"))
    search_vectors, projection_meta = project_vectors_for_search(vectors, config)
    config.vector_dimensions = int(search_vectors.shape[1])
    print(f"[Stage 6] Ensuring Azure AI Search index: {config.index_name}")
    ensure_store_vector_index(config)

    documents = build_store_vector_documents(
        store_ids,
        search_vectors,
        dim_store,
        store_dna_index,
        fusion_method=str(manifest.get("fusion_method", "")),
    )
    print(f"[Stage 6] Uploading {len(documents):,} store vectors...")
    uploaded = upload_store_vector_documents(config, documents)
    count = get_store_vector_index_count(config)
    np.savez_compressed(
        output_dir / "store_dna_search_vectors.npz",
        store_ids=np.array(store_ids),
        vectors=search_vectors,
    )

    result = {
        "search_index": config.index_name,
        "vector_dimensions": int(search_vectors.shape[1]),
        "projection": projection_meta,
        "stores_loaded": len(store_ids),
        "documents_uploaded": uploaded,
        "index_document_count": count,
        "elapsed_sec": round(time.time() - start_time, 1),
    }
    (output_dir / "store_vector_index_manifest.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"[Stage 6] Wrote outputs to {output_dir}")
    print(
        f"[Stage 6] Complete: uploaded {uploaded} docs to {config.index_name} "
        f"in {result['elapsed_sec']}s"
    )
    return result
