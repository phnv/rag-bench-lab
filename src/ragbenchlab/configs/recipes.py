"""
Hardcoded Recipe catalogue for the RAG Bench Lab project.

A Recipe fully specifies one RAG pipeline configuration — one choice per
stage.  All Recipes used across Phase 1–3 are defined here.  Adding a new
Recipe is one new entry in RECIPES; no other file needs to change.

Phase 1 Recipes
---------------
R0  No RAG baseline
R1  BM25 only
R2  Dense only  (sentence-transformers/all-MiniLM-L6-v2 + FAISS)
R3  Hybrid BM25 + Dense with RRF fusion
R4  Hybrid + Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)

Phase 2+ stubs are commented out and will be uncommented as implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Stage enums
# ---------------------------------------------------------------------------


class RetrievalStrategy(str, Enum):
    """Retrieval mechanism for a Recipe."""

    NONE = "none"      # R0: No RAG — answer from LLM only
    BM25 = "bm25"      # Lexical (rank_bm25)
    DENSE = "dense"    # Vector / semantic (sentence-transformers + FAISS)
    HYBRID = "hybrid"  # BM25 + Dense concurrently → fused with RRF


class ContextStrategy(str, Enum):
    """Context selection strategy applied after retrieval / reranking."""

    TOP_K = "top_k"
    # Phase 2 stubs — uncomment when implemented:
    # ADJACENT = "adjacent"
    # PARENT = "parent"
    # DEDUP = "deduplication"
    # COMPRESSION = "compression"


class QueryTransform(str, Enum):
    """Query transformation applied before retrieval."""

    ORIGINAL = "original"   # Passthrough — no transformation
    # Phase 2 stubs — uncomment when implemented:
    # MULTI_QUERY = "multi_query"
    # HYDE = "hyde"
    # STEP_BACK = "step_back"
    # DECOMPOSITION = "decomposition"


# ---------------------------------------------------------------------------
# RecipeConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeConfig:
    """Fully-specified configuration for one RAG pipeline Recipe."""

    recipe_id: str
    """Short identifier (e.g. 'R0', 'R1').  Used in output paths."""

    description: str
    """Human-readable description for logging and the dashboard."""

    retrieval: RetrievalStrategy
    """The retrieval mechanism to use."""

    reranker: bool = False
    """Whether to apply a cross-encoder reranker after retrieval."""

    top_k: int = 10
    """Number of candidate documents to retrieve (before reranking)."""

    query_transform: QueryTransform = QueryTransform.ORIGINAL
    """Query transformation strategy applied before retrieval."""

    fusion: bool = False
    """Whether RRF fusion is applied (required when retrieval=HYBRID)."""

    context_strategy: ContextStrategy = ContextStrategy.TOP_K
    """How the final context window is constructed from ranked candidates."""


# ---------------------------------------------------------------------------
# RECIPES registry  (Phase 1: R0–R4 baseline ladder)
# ---------------------------------------------------------------------------

RECIPES: dict[str, RecipeConfig] = {
    "R0": RecipeConfig(
        recipe_id="R0",
        description="No RAG baseline — LLM answers without retrieved context",
        retrieval=RetrievalStrategy.NONE,
    ),
    "R1": RecipeConfig(
        recipe_id="R1",
        description="BM25 only — lexical retrieval with rank_bm25",
        retrieval=RetrievalStrategy.BM25,
    ),
    "R2": RecipeConfig(
        recipe_id="R2",
        description="Dense only — sentence-transformers/all-MiniLM-L6-v2 embeddings + FAISS",
        retrieval=RetrievalStrategy.DENSE,
    ),
    "R3": RecipeConfig(
        recipe_id="R3",
        description="Hybrid BM25 + Dense with RRF fusion (k=60)",
        retrieval=RetrievalStrategy.HYBRID,
        fusion=True,
    ),
    "R4": RecipeConfig(
        recipe_id="R4",
        description="Hybrid + Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)",
        retrieval=RetrievalStrategy.HYBRID,
        reranker=True,
        fusion=True,
    ),
    # Phase 2+ Recipes will be added here after the decision tree selects them.
}


def get_recipe(recipe_id: str) -> RecipeConfig:
    """Return the RecipeConfig for *recipe_id*, raising KeyError on miss."""
    try:
        return RECIPES[recipe_id]
    except KeyError:
        valid = ", ".join(sorted(RECIPES))
        raise KeyError(
            f"Unknown recipe_id {recipe_id!r}. Valid IDs: {valid}"
        ) from None
