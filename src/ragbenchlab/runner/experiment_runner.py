"""
ExperimentRunner — the single orchestration point for all RAG experiments.

Usage (from main.py)
--------------------
    runner = ExperimentRunner()
    out_dir = runner.run("R0", "nq", mode="dev")

Recipe resolution flow
----------------------
    recipe_id  ──► get_recipe()     ──► RecipeConfig
    scenario_id ──► get_scenario()  ──► ScenarioConfig
                     │
                     ▼
              build_pipeline()      ──► (IngestPipeline, InferencePipeline)
                     │
                     ▼
              RagWorkbench Experiment.run()
                     │
                     ▼
              output/{scenario_id}/{recipe_id}/{mode}_n{limit}_s{seed}_{timestamp}/

The ExperimentRunner is the ONLY place that knows which recipe_id maps to
which concrete pipeline class.  All parameters come from RecipeConfig.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

from ragworkbench import DataLoaderFactory, Experiment
from ragworkbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragworkbench.boards.board_model import ExperimentConfig
from ragworkbench.api.ingest import IngestPipeline
from ragworkbench.api.inference import InferencePipeline
from ragworkbench.eval import load_metric_definitions

from ragbenchlab.configs.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    K_CONTEXT,
    K_RETRIEVE,
    RERANKER_MODEL,
    RRF_K,
    ScenarioConfig,
    get_scenario,
)
from ragbenchlab.configs.recipes import RecipeConfig, RetrievalStrategy, get_recipe
from ragbenchlab.pipelines.stub import StubIngestPipeline, StubInferencePipeline
from ragbenchlab.pipelines.bm25 import BM25IngestPipeline, BM25InferencePipeline
from ragbenchlab.pipelines.dense import DenseIngestPipeline, DenseInferencePipeline
from ragbenchlab.pipelines.hybrid import HybridIngestPipeline, HybridInferencePipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

# Maps recipe_id → (IngestPipeline class, InferencePipeline class).
# As real implementations are added in M4–M5, replace stub entries here.
_PIPELINE_REGISTRY: dict[str, tuple[type, type]] = {
    "R0": (StubIngestPipeline, StubInferencePipeline),
    "R1": (BM25IngestPipeline, BM25InferencePipeline),
    "R2": (DenseIngestPipeline, DenseInferencePipeline),
    "R3": (HybridIngestPipeline, HybridInferencePipeline),
    "R4": (HybridIngestPipeline, HybridInferencePipeline),
}


def build_pipeline(
    recipe_id: str,
    scenario_id: str,
) -> Tuple[IngestPipeline, InferencePipeline]:
    """
    Construct and return an (IngestPipeline, InferencePipeline) pair for
    *recipe_id*, fully configured from the matching RecipeConfig.

    Parameters
    ----------
    recipe_id : str
        Must be a key in RECIPES (e.g. 'R0', 'R1', …).

    Returns
    -------
    tuple[IngestPipeline, InferencePipeline]
    """
    recipe = get_recipe(recipe_id)

    if recipe_id not in _PIPELINE_REGISTRY:
        raise NotImplementedError(
            f"No pipeline implementation registered for recipe_id={recipe_id!r}. "
            "Add an entry to _PIPELINE_REGISTRY in experiment_runner.py."
        )

    IngestClass, InferClass = _PIPELINE_REGISTRY[recipe_id]
    return IngestClass(recipe, scenario_id), InferClass(recipe, scenario_id)


# ---------------------------------------------------------------------------
# Metadata & model helpers
# ---------------------------------------------------------------------------


def resolve_models_for_recipe(
    recipe: RecipeConfig,
) -> tuple[dict[str, str | None], list[str]]:
    """
    Determine role-by-role models and the deduplicated active local model list
    for *recipe*.

    Returns
    -------
    tuple[dict[str, str | None], list[str]]
        (models_by_role, active_local_models)
    """
    uses_retrieval = recipe.retrieval != RetrievalStrategy.NONE
    uses_dense = recipe.retrieval in (
        RetrievalStrategy.DENSE,
        RetrievalStrategy.HYBRID,
    )

    tokenizer = EMBEDDING_MODEL if uses_retrieval else None
    embedding_model = EMBEDDING_MODEL if uses_dense else None
    reranker_model = RERANKER_MODEL if recipe.reranker else None
    generation_model: str | None = None

    models_by_role: dict[str, str | None] = {
        "tokenizer": tokenizer,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "generation_model": generation_model,
    }

    active_local_models: list[str] = []
    for model_name in (tokenizer, embedding_model, reranker_model):
        if model_name is not None and model_name not in active_local_models:
            active_local_models.append(model_name)

    return models_by_role, active_local_models


def build_experiment_metadata(
    experiment_id: str,
    timestamp_utc: str,
    mode: str,
    scenario: ScenarioConfig,
    recipe: RecipeConfig,
    question_limit: int,
    description: str = "",
    comments: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build the comprehensive metadata dictionary written to
    ``experiment_metadata_{experiment_id}.json``.
    """
    models_by_role, _ = resolve_models_for_recipe(recipe)
    dataset_val = (
        scenario.dataset_name.value
        if hasattr(scenario.dataset_name, "value")
        else str(scenario.dataset_name)
    )

    return {
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp_utc,
        "mode": mode,
        "description": description,
        "comments": list(comments) if comments else [],
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "dataset_name": dataset_val,
            "question_limit": question_limit,
            "seed": scenario.seed,
            "description": scenario.description,
        },
        "recipe": {
            "recipe_id": recipe.recipe_id,
            "description": recipe.description,
            "retrieval": recipe.retrieval.value,
            "reranker": recipe.reranker,
            "top_k": recipe.top_k,
            "query_transform": recipe.query_transform.value,
            "fusion": recipe.fusion,
            "context_strategy": recipe.context_strategy.value,
        },
        "retrieval_params": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "k_retrieve": K_RETRIEVE,
            "k_context": K_CONTEXT,
            "rrf_k": RRF_K,
        },
        "models": models_by_role,
    }


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """
    Orchestrates a single Experiment run from recipe_id + scenario_id.

    Parameters
    ----------
    output_root : Path | str
        Root directory for all experiment outputs.  Defaults to 'output/'.
    cache_root : Path | str
        Root directory for RagWorkbench caching.  Defaults to 'cache/'.
    """

    def __init__(
        self,
        output_root: Path | str = Path("output"),
        cache_root: Path | str = Path("cache"),
    ) -> None:
        self._output_root = Path(output_root)
        self._cache_root = Path(cache_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        recipe_id: str,
        scenario_id: str,
        mode: str = "dev",
        description: str = "",
        comments: list[str] | None = None,
    ) -> Path:
        """
        Execute one Experiment and return the run-specific output directory path.

        Parameters
        ----------
        recipe_id : str
            Which Recipe to run (e.g. 'R0', 'R1').
        scenario_id : str
            Which Scenario dataset to use (e.g. 'nq', 'hotpotqa').
        mode : "dev" | "eval"
            Whether to use the small dev subset or the frozen eval subset.
        description : str
            Optional free-text description for this experiment run.
        comments : list[str] | None
            Optional list of short comment strings for this experiment run.

        Returns
        -------
        Path
            The directory where results were written:
            output/{scenario_id}/{recipe_id}/{mode}_n{limit}_s{seed}_{YYYYMMDD_HHMMSS}/
        """
        if mode not in ("dev", "eval"):
            raise ValueError(f"mode must be 'dev' or 'eval', got {mode!r}")

        scenario: ScenarioConfig = get_scenario(scenario_id)
        recipe: RecipeConfig = get_recipe(recipe_id)
        question_limit = (
            scenario.eval_size if mode == "eval" else scenario.dev_size
        )

        now_utc = datetime.now(timezone.utc)
        timestamp_slug = now_utc.strftime("%Y%m%d_%H%M%S")
        timestamp_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            "Starting experiment  recipe=%s  scenario=%s  mode=%s  n=%d  seed=%d",
            recipe_id,
            scenario_id,
            mode,
            question_limit,
            scenario.seed,
        )

        # 1 & 2. Load dataset with deterministic subset sampling
        sampling = DataSamplingParams(
            question_limit=question_limit,
            seed=scenario.seed,
        )
        data_loader = DataLoaderFactory.create_loader(
            scenario.dataset_name,
            sampling_params=sampling,
            cache_dir=self._cache_root,
        )
        logger.info(
            "Loaded data with sampling params for scenario=%s mode=%s (seed=%d)",
            scenario_id,
            mode,
            scenario.seed,
        )

        # 3. Build pipeline pair
        ingest_pipeline, inference_pipeline = build_pipeline(recipe_id, scenario_id)

        # 4. Build experiment_id and prepare collision-free output subfolder
        experiment_id = (
            f"{scenario_id}__{recipe_id}__{mode}_n{question_limit}_s{scenario.seed}"
        )
        run_folder_name = (
            f"{mode}_n{question_limit}_s{scenario.seed}_{timestamp_slug}"
        )
        out_dir = self._output_root / scenario_id / recipe_id / run_folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Setup evaluation metrics for retrieval phase
        defs = load_metric_definitions()
        eval_metrics = [
            defs.get_metric_definition("unitxt.context_correctness.retrieval_at_k"),
            defs.get_metric_definition("unitxt.context_correctness.mrr"),
            defs.get_metric_definition("unitxt.context_correctness.map"),
        ]

        # 5. Run via RagWorkbench Experiment
        experiment = Experiment(
            experiment_id=experiment_id,
            data_loader=data_loader,
            ingest_pipeline=ingest_pipeline,
            inference_pipeline=inference_pipeline,
            eval_metrics=eval_metrics,
            experiment_config=ExperimentConfig(),
            cache_dir=self._cache_root,
        )
        result = experiment.run()

        # 6. Populate cost_data.models_used with active local models
        _, active_local_models = resolve_models_for_recipe(recipe)
        if result.cost_data is not None:
            existing_models = list(result.cost_data.models_used or [])
            for m in active_local_models:
                if m not in existing_models:
                    existing_models.append(m)
            result.cost_data.models_used = existing_models

        # 7. Export results (.csv and .json)
        result.export_all(
            output_path=out_dir,
            config_seq=0,
            dataset_seq=0,
            config_name=recipe_id,
            dataset_id=scenario_id,
        )

        # 8. Write 3rd file: experiment_metadata_{experiment_id}.json
        metadata = build_experiment_metadata(
            experiment_id=experiment_id,
            timestamp_utc=timestamp_iso,
            mode=mode,
            scenario=scenario,
            recipe=recipe,
            question_limit=question_limit,
            description=description,
            comments=comments,
        )
        metadata_path = out_dir / f"experiment_metadata_{experiment_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Exported experiment metadata to %s", metadata_path)

        logger.info("Experiment complete — results in %s", out_dir)
        return out_dir
