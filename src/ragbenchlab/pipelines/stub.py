"""
Stub pipeline implementations for M3 dataset-setup validation.

These stubs let ExperimentRunner wire up the full run loop so that
deterministic subset sampling, output path creation, and the RagWorkbench
Experiment lifecycle can all be exercised without real retrieval logic.

Real implementations replace these stubs in M4 (BM25, Dense, Hybrid) and
M5 (reranker).  Each real Recipe will subclass IngestPipeline +
InferencePipeline in its own file under src/ragbenchlab/pipelines/.
"""

from __future__ import annotations

import logging
from typing import Any, List

from ragworkbench.api.ingest import IngestArtifact, IngestParams, IngestPipeline
from ragworkbench.api.inference import InferenceParams, InferencePipeline, InferenceResult

from ragbenchlab.configs.recipes import RecipeConfig

logger = logging.getLogger(__name__)


class StubIngestPipeline(IngestPipeline):
    """
    No-op ingest pipeline used during M3 to validate the run harness.

    Does not build any index; simply records the corpus document count
    so the InferencePipeline can confirm corpus was passed through.
    """

    def __init__(self, recipe: RecipeConfig, scenario_id: str) -> None:
        super().__init__(IngestParams())
        self._recipe = recipe
        self.scenario_id = scenario_id

    def process(self, data_loader: Any, **kwargs) -> list[IngestArtifact]:
        corpus = data_loader.get_corpus()
        doc_count = len(corpus)
        logger.info(
            "[%s] StubIngestPipeline: indexed %d corpus documents (no-op)",
            self._recipe.recipe_id,
            doc_count,
        )
        return [IngestArtifact(data={"doc_count": doc_count, "recipe_id": self._recipe.recipe_id})]


class StubInferencePipeline(InferencePipeline):
    """
    No-op inference pipeline used during M3 to validate the run harness.

    Returns a placeholder answer with the question echoed so that output
    files are written and can be inspected for correctness of the pipeline.
    """

    def __init__(self, recipe: RecipeConfig, scenario_id: str) -> None:
        super().__init__(InferenceParams())
        self._recipe = recipe
        self.scenario_id = scenario_id

    def set_ingest_artifacts(self, ingest_artifacts: List[IngestArtifact]) -> None:
        # Nothing to store — stubs carry no real index.
        pass

    def process_no_cache(self, benchmark_entry: Any, **kwargs) -> InferenceResult:
        return InferenceResult(
            question_id=benchmark_entry.question_id,
            question=benchmark_entry.question,
            answer=f"[STUB:{self._recipe.recipe_id}] No answer — stub pipeline",
            context=["<stub context>"],
            metadata={
                "recipe_id": self._recipe.recipe_id,
                "retrieval": self._recipe.retrieval.value,
                "stub": True,
            },
        )
