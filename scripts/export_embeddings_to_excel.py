"""
Export row-level or final store-level vector documents from Azure AI Search.

Usage (from project root):
    python -m scripts.export_embeddings_to_excel
    python -m scripts.export_embeddings_to_excel --modality reviews
    python -m scripts.export_embeddings_to_excel --with-vector
    python -m scripts.export_embeddings_to_excel --format excel
    python -m scripts.export_embeddings_to_excel --index store
    python -m scripts.export_embeddings_to_excel --index store --vector-preview 10

Notes:
- By default the large content_vector is NOT exported (keeps the file readable).
  Use --with-vector to include it (as a JSON string in one cell), or
  --vector-preview N to include only the first N numbers.
- Output is written to data/USA_100_Stores/exports/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from src.store_dna_search import SearchIndexConfig
from src.store_dna_vector_index import StoreVectorIndexConfig

ALL_MODALITIES = ("reviews", "news", "reports", "products", "ops_weekly")
ROW_META_FIELDS = [
    "id",
    "store_id",
    "modality",
    "source_id",
    "retailer",
    "content",
    "embedding_model",
    "embedding_dim",
]
STORE_META_FIELDS = [
    "id",
    "store_id",
    "retailer",
    "banner",
    "store_name",
    "city",
    "state",
    "store_format",
    "fusion_method",
    "store_dna_dim",
    "store_dna_norm",
]


def fetch_all_documents(
    endpoint: str,
    api_key: str,
    index_name: str,
    select: list[str],
    modality: str | None = None,
    page_size: int = 500,
) -> list[dict]:
    """Paginate the whole index (or one modality) and return the documents."""
    client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(api_key),
    )
    documents: list[dict] = []
    skip = 0
    while True:
        page = list(
            client.search(
                search_text="*",
                filter=f"modality eq '{modality}'" if modality else None,
                select=select,
                skip=skip,
                top=page_size,
            )
        )
        if not page:
            break
        documents.extend(page)
        if len(page) < page_size:
            break
        skip += page_size
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Azure Search embeddings to CSV/Excel")
    parser.add_argument(
        "--index",
        choices=("rows", "store"),
        default="rows",
        help="Export Stage 3 row documents or final store-level documents",
    )
    parser.add_argument("--modality", choices=ALL_MODALITIES, help="Export only one modality")
    parser.add_argument("--with-vector", action="store_true", help="Include the full vector as JSON")
    parser.add_argument("--vector-preview", type=int, default=0, help="Include first N vector values only")
    parser.add_argument("--format", choices=("csv", "excel"), default="csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")
    if args.index == "store":
        if args.modality:
            parser.error("--modality is only valid with --index rows")
        store_config = StoreVectorIndexConfig.from_env(project_root)
        endpoint = store_config.search_endpoint
        api_key = store_config.search_api_key
        index_name = store_config.index_name
        meta_fields = STORE_META_FIELDS
        vector_field = "store_dna_vector"
    else:
        row_config = SearchIndexConfig.from_env()
        endpoint = row_config.endpoint
        api_key = row_config.api_key
        index_name = row_config.index_name
        meta_fields = ROW_META_FIELDS
        vector_field = "content_vector"

    select = list(meta_fields)
    if args.with_vector or args.vector_preview:
        select.append(vector_field)

    print(f"[Export] Index: {index_name}")
    documents = fetch_all_documents(
        endpoint,
        api_key,
        index_name,
        select,
        modality=args.modality,
    )
    print(f"[Export] Retrieved {len(documents):,} documents")

    rows: list[dict] = []
    for doc in documents:
        row = {field: doc.get(field) for field in meta_fields}
        vector = doc.get(vector_field)
        if args.with_vector and vector is not None:
            row[vector_field] = json.dumps(vector)
        elif args.vector_preview and vector is not None:
            row[f"{vector_field}_preview"] = json.dumps(list(vector)[: args.vector_preview])
        rows.append(row)

    frame = pd.DataFrame(rows)
    export_dir = project_root / "data" / "USA_100_Stores" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.modality or ("store_vectors" if args.index == "store" else "all")

    if args.format == "excel":
        out_path = export_dir / f"embeddings_{suffix}.xlsx"
        frame.to_excel(out_path, index=False)
    else:
        out_path = export_dir / f"embeddings_{suffix}.csv"
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[Export] Wrote {len(frame):,} rows to {out_path}")


if __name__ == "__main__":
    main()
