"""
Stage 5 - StoreDNA Builder for Retail StoreDNA.

Reads Stage 4 modality vectors, applies weighted late fusion, and writes one
store-level fingerprint vector per store for downstream indexing and analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EMBEDDING_MODALITIES = ("reviews", "news", "reports", "products", "ops_weekly")
STRUCTURED_MODALITY = "structured_ops"
ALL_MODALITIES = EMBEDDING_MODALITIES + (STRUCTURED_MODALITY,)
DEFAULT_WEIGHTS = {
    "reviews": 1.0,
    "news": 1.0,
    "reports": 1.0,
    "products": 1.0,
    "ops_weekly": 1.0,
    "structured_ops": 0.5,
}


@dataclass
class StoreDNABuilderConfig:
    """Stage 5 settings."""

    max_stores: int | None = None
    l2_normalize_inputs: bool = True
    l2_normalize_output: bool = True
    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def load_npz_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    """Load one Stage 4 vector file."""
    data = np.load(path)
    store_ids = [str(x) for x in data["store_ids"]]
    vectors = data["vectors"].astype(np.float32)
    return store_ids, vectors


def load_stage4_vectors(modality_dir: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    """Load all Stage 4 modality vector files."""
    vectors: dict[str, tuple[list[str], np.ndarray]] = {}
    for modality in ALL_MODALITIES:
        path = modality_dir / f"{modality}_vectors.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 4 output: {path}")
        vectors[modality] = load_npz_vectors(path)
    return vectors


def align_modalities(
    loaded: dict[str, tuple[list[str], np.ndarray]],
    store_ids: list[str],
) -> dict[str, np.ndarray]:
    """Align all modality matrices to the same store ordering."""
    aligned: dict[str, np.ndarray] = {}
    for modality, (modality_store_ids, vectors) in loaded.items():
        index = {sid: i for i, sid in enumerate(modality_store_ids)}
        matrix = np.zeros((len(store_ids), vectors.shape[1]), dtype=np.float32)
        for row_idx, sid in enumerate(store_ids):
            if sid in index:
                matrix[row_idx] = vectors[index[sid]]
        aligned[modality] = matrix
    return aligned


def weighted_late_fusion(
    aligned: dict[str, np.ndarray],
    weights: dict[str, float],
    l2_normalize_inputs: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Scale each modality and concatenate into one fused vector."""
    blocks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for modality in ALL_MODALITIES:
        matrix = aligned[modality]
        if l2_normalize_inputs and len(matrix):
            matrix = l2_normalize(matrix)
        weight = float(weights.get(modality, 1.0))
        weighted = matrix * weight
        blocks.append(weighted)
        metadata.append(
            {
                "modality": modality,
                "vector_dim": int(matrix.shape[1]),
                "weight": weight,
            }
        )

    fused = np.concatenate(blocks, axis=1).astype(np.float32)
    return fused, metadata


def build_store_dna(
    modality_dir: Path,
    output_dir: Path,
    config: StoreDNABuilderConfig,
) -> dict[str, Any]:
    """Full Stage 5 pipeline."""
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_stage4_vectors(modality_dir)
    base_store_ids = loaded["reviews"][0]
    if config.max_stores:
        base_store_ids = base_store_ids[: config.max_stores]
        loaded = {
            modality: (
                store_ids[: config.max_stores],
                vectors[: config.max_stores],
            )
            for modality, (store_ids, vectors) in loaded.items()
        }
        print(f"[Stage 5] Store filter applied: {len(base_store_ids)} stores")
    else:
        print(f"[Stage 5] Processing {len(base_store_ids)} stores")

    aligned = align_modalities(loaded, base_store_ids)
    fused, modality_meta = weighted_late_fusion(
        aligned,
        config.weights,
        l2_normalize_inputs=config.l2_normalize_inputs,
    )
    if config.l2_normalize_output and len(fused):
        fused = l2_normalize(fused)

    np.savez_compressed(
        output_dir / "store_dna_vectors.npz",
        store_ids=np.array(base_store_ids),
        vectors=fused,
    )

    row_metadata: list[dict[str, Any]] = []
    for row_idx, store_id in enumerate(base_store_ids):
        row = {
            "store_id": store_id,
            "store_dna_dim": int(fused.shape[1]),
            "store_dna_norm": float(np.linalg.norm(fused[row_idx])),
        }
        for modality in ALL_MODALITIES:
            modality_matrix = aligned[modality]
            row[f"{modality}_present"] = bool(np.any(modality_matrix[row_idx] != 0))
            row[f"{modality}_norm"] = float(np.linalg.norm(modality_matrix[row_idx]))
        row_metadata.append(row)

    metadata_df = pd.DataFrame(row_metadata)
    metadata_df.to_csv(output_dir / "store_dna_index.csv", index=False)

    manifest = {
        "fusion_method": "weighted_late_fusion_concat",
        "l2_normalize_inputs": config.l2_normalize_inputs,
        "l2_normalize_output": config.l2_normalize_output,
        "weights": {k: float(v) for k, v in config.weights.items()},
        "store_count": len(base_store_ids),
        "store_dna_dim": int(fused.shape[1]),
        "modalities": modality_meta,
        "outputs": {
            "vectors": "store_dna_vectors.npz",
            "index": "store_dna_index.csv",
        },
        "elapsed_sec": round(time.time() - start_time, 1),
    }
    (output_dir / "store_dna_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"[Stage 5] Wrote outputs to {output_dir}")
    print(
        f"[Stage 5] Complete: {manifest['store_count']} stores, "
        f"dim={manifest['store_dna_dim']} in {manifest['elapsed_sec']}s"
    )
    return manifest
