"""
FastAPI backend for Front end.

KPIs from curated CSV data; peer similarity from Stage 6 store embeddings.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.store_dna_frontend_api import (  # noqa: E402
    compare_stores,
    get_cache,
    get_health,
    get_portfolio_payload,
    get_store,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        get_cache()
    except Exception as exc:  # noqa: BLE001 — surface on /api/health instead of crashing startup
        print(f"[Front end API] Cache warmup deferred: {exc}")
    yield


app = FastAPI(title="Retail Store DNA Front End API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    try:
        return get_health()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/stores")
def stores() -> dict:
    try:
        return get_portfolio_payload()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/stores/{store_id}")
def store_detail(store_id: str) -> dict:
    store = get_store(store_id)
    if store is None:
        raise HTTPException(status_code=404, detail=f"Store not found: {store_id}")
    return store


@app.get("/api/compare")
def compare(
    store_a: str = Query(...),
    store_b: str = Query(...),
) -> dict:
    if store_a == store_b:
        return {"store_a": store_a, "store_b": store_b, "similarity": 1.0}
    result = compare_stores(store_a, store_b)
    if result["store_a_doc"] is None or result["store_b_doc"] is None:
        raise HTTPException(status_code=404, detail="One or both stores were not found")
    return result
