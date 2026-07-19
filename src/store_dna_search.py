"""
Azure AI Search — index and upload row-level StoreDNA embeddings (Stage 3).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

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


@dataclass
class SearchIndexConfig:
    endpoint: str
    api_key: str
    index_name: str
    vector_dimensions: int = 3072
    upload_batch_size: int = 500

    @classmethod
    def from_env(cls) -> SearchIndexConfig:
        import os

        return cls(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/"),
            api_key=os.environ["AZURE_SEARCH_API_KEY"],
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "store-dna-embeddings"),
            vector_dimensions=int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "3072")),
            upload_batch_size=int(os.getenv("AZURE_SEARCH_UPLOAD_BATCH_SIZE", "500")),
        )


def sanitize_search_document_id(raw_id: str) -> str:
    """Azure Search document keys: letters, digits, _, -, = only."""
    safe = re.sub(r"[^a-zA-Z0-9_\-=]", "_", raw_id)
    return safe[:1024]


def ensure_embedding_index(config: SearchIndexConfig) -> None:
    """Create or update the vector search index for row-level embeddings."""
    index_client = SearchIndexClient(
        endpoint=config.endpoint,
        credential=AzureKeyCredential(config.api_key),
    )

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-default")],
        profiles=[
            VectorSearchProfile(
                name="embedding-profile",
                algorithm_configuration_name="hnsw-default",
            )
        ],
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="store_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="modality", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="retailer", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="embedding_model", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="embedding_dim", type=SearchFieldDataType.Int32, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=True,
            vector_search_dimensions=config.vector_dimensions,
            vector_search_profile_name="embedding-profile",
        ),
    ]

    index = SearchIndex(
        name=config.index_name,
        fields=fields,
        vector_search=vector_search,
    )
    index_client.create_or_update_index(index)


def build_search_documents(
    modality: str,
    enriched: pd.DataFrame,
    embeddings: np.ndarray,
    id_col: str,
    embed_texts: list[str],
    retailer_map: dict[str, str],
    embedding_model: str,
) -> list[dict]:
    """Map enriched rows + vectors to Azure AI Search documents."""
    documents: list[dict] = []
    for i in range(len(enriched)):
        row = enriched.iloc[i]
        source_id = str(row[id_col])
        store_id = str(row["store_id"]) if "store_id" in row.index and pd.notna(row["store_id"]) else ""
        doc_id = sanitize_search_document_id(f"{modality}_{source_id}")
        documents.append({
            "id": doc_id,
            "store_id": store_id,
            "modality": modality,
            "source_id": source_id,
            "retailer": retailer_map.get(store_id, ""),
            "content": embed_texts[i],
            "content_vector": embeddings[i].astype(float).tolist(),
            "embedding_model": embedding_model,
            "embedding_dim": int(embeddings.shape[1]),
        })
    return documents


def upload_embedding_documents(
    config: SearchIndexConfig,
    documents: list[dict],
    pause_sec: float = 0.2,
) -> int:
    """Merge-or-upload documents to Azure AI Search in batches."""
    if not documents:
        return 0

    search_client = SearchClient(
        endpoint=config.endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.api_key),
    )

    uploaded = 0
    for start in range(0, len(documents), config.upload_batch_size):
        batch = documents[start : start + config.upload_batch_size]
        search_client.merge_or_upload_documents(batch)
        uploaded += len(batch)
        if start + config.upload_batch_size < len(documents):
            time.sleep(pause_sec)

    return uploaded


def get_index_document_count(config: SearchIndexConfig) -> int:
    search_client = SearchClient(
        endpoint=config.endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.api_key),
    )
    return search_client.get_document_count()


def fetch_documents_by_modality(
    config: SearchIndexConfig,
    modality: str,
    select: list[str] | None = None,
    page_size: int = 500,
    max_retries: int = 5,
) -> list[dict]:
    """Return all index documents for a modality (paginated, with retries)."""
    search_client = SearchClient(
        endpoint=config.endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.api_key),
    )
    fields = select or ["store_id", "content_vector"]
    documents: list[dict] = []
    skip = 0
    while True:
        for attempt in range(max_retries):
            try:
                page = list(
                    search_client.search(
                        search_text="*",
                        filter=f"modality eq '{modality}'",
                        select=fields,
                        skip=skip,
                        top=page_size,
                    )
                )
                break
            except Exception as exc:
                if attempt + 1 >= max_retries:
                    raise
                wait = 2 ** attempt
                print(f"[Search] {modality} skip={skip}: retry {attempt + 1}/{max_retries} ({exc})")
                time.sleep(wait)
        if not page:
            break
        documents.extend(page)
        if len(page) < page_size:
            break
        skip += page_size
        time.sleep(0.1)
    return documents
