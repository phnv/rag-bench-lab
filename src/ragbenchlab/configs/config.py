"""
Global hardcoded parameters for the RAG Bench Lab project.

All constants that affect reproducibility live here so there is a single
source of truth.  To change subset sizes for a full run, edit DEV_SIZE and
EVAL_SIZE; all downstream code picks up the new values automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragworkbench import DatasetName

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED: int = 42

# ---------------------------------------------------------------------------
# Subset sizes  (placeholder — change to 30 / 100 before final runs)
# ---------------------------------------------------------------------------

DEV_SIZE: int = 5
EVAL_SIZE: int = 10

K_RETRIEVE: int = 50        # candidates fetched per retriever before context slicing
K_CONTEXT: int = 10         # candidates kept for context (= RecipeConfig.top_k)

CHUNK_SIZE: int = 200       # tokens per chunk (fits all-MiniLM-L6-v2's 256-token max)
CHUNK_OVERLAP: int = 25     # token overlap between consecutive chunks
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K: int = 60             # RRF rank constant (ADR-0004); logged with every fusion call

# ---------------------------------------------------------------------------
# ScenarioConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioConfig:
    """Fully-specified dataset / sampling configuration for one Scenario."""

    scenario_id: str
    """Short identifier used in output paths (e.g. 'nq', 'hotpotqa')."""

    dataset_name: DatasetName
    """The RagWorkbench enum key for this dataset."""

    dev_size: int = DEV_SIZE
    """Number of questions in the *dev* subset (scratch-pad, not frozen)."""

    eval_size: int = EVAL_SIZE
    """Number of questions in the *eval* subset (frozen, comparative)."""

    seed: int = SEED
    """Fixed random seed used for deterministic sampling."""

    description: str = ""
    """Human-readable description of the scenario."""


# ---------------------------------------------------------------------------
# Four canonical Scenarios  (ADR-0003)
#
# WatsonX DocsQA has three corpus variants (TXT / HTML / MD).
# We use TXT as the primary — plain-text is the most retrieval-friendly
# format and avoids HTML/Markdown token noise during BM25 and dense indexing.
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, ScenarioConfig] = {
    "nq": ScenarioConfig(
        scenario_id="nq",
        dataset_name=DatasetName.CLAP_NQ,
        description="Wikipedia × Single-hop — Natural Questions (CLAP-NQ variant)",
    ),
    "hotpotqa": ScenarioConfig(
        scenario_id="hotpotqa",
        dataset_name=DatasetName.HOTPOT_QA,
        description="Wikipedia × Multi-hop — HotpotQA (2-document reasoning chains)",
    ),
    "watsonx": ScenarioConfig(
        scenario_id="watsonx",
        dataset_name=DatasetName.WATSONX_DOCS_QA_TXT,
        description="Technical × Single-hop — WatsonX Docs QA (plain-text corpus)",
    ),
    "dabstep": ScenarioConfig(
        scenario_id="dabstep",
        dataset_name=DatasetName.DABSTEP,
        description="Technical × Multi-hop — DABStep (code / multi-step reasoning)",
    ),
}


def get_scenario(scenario_id: str) -> ScenarioConfig:
    """Return the ScenarioConfig for *scenario_id*, raising KeyError on miss."""
    try:
        return SCENARIOS[scenario_id]
    except KeyError:
        valid = ", ".join(sorted(SCENARIOS))
        raise KeyError(
            f"Unknown scenario_id {scenario_id!r}. Valid IDs: {valid}"
        ) from None
