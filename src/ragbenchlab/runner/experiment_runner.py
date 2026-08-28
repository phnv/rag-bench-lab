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
              output/{scenario_id}/{recipe_id}/

The ExperimentRunner is the ONLY place that knows which recipe_id maps to
which concrete pipeline class.  All parameters come from RecipeConfig.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from ragworkbench import DataLoaderFactory, Experiment
from ragworkbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragworkbench.boards.board_model import ExperimentConfig
from ragworkbench.api.ingest import IngestPipeline
from ragworkbench.api.inference import InferencePipeline
from ragworkbench.eval import load_metric_definitions

from ragbenchlab.configs.config import ScenarioConfig, get_scenario
from ragbenchlab.configs.recipes import RecipeConfig, get_recipe
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
    ) -> Path:
        """
        Execute one Experiment and return the output directory path.

        Parameters
        ----------
        recipe_id : str
            Which Recipe to run (e.g. 'R0', 'R1').
        scenario_id : str
            Which Scenario dataset to use (e.g. 'nq', 'hotpotqa').
        mode : "dev" | "eval"
            Whether to use the small dev subset or the frozen eval subset.

        Returns
        -------
        Path
            The directory where results were written:
            output/{scenario_id}/{recipe_id}/
        """
        if mode not in ("dev", "eval"):
            raise ValueError(f"mode must be 'dev' or 'eval', got {mode!r}")

        scenario: ScenarioConfig = get_scenario(scenario_id)
        recipe: RecipeConfig = get_recipe(recipe_id)

        logger.info(
            "Starting experiment  recipe=%s  scenario=%s  mode=%s",
            recipe_id,
            scenario_id,
            mode,
        )

        # 1 & 2. Load dataset with deterministic subset sampling
        sampling = DataSamplingParams(
            question_limit=scenario.eval_size if mode == "eval" else scenario.dev_size,
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

        # 4. Prepare output directory
        out_dir = self._output_root / scenario_id / recipe_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # 5. Build experiment_id that encodes all run parameters
        experiment_id = f"{scenario_id}__{recipe_id}__{mode}"

        # Setup evaluation metrics for retrieval phase
        defs = load_metric_definitions()
        eval_metrics = [
            defs.get_metric_definition("unitxt.context_correctness.retrieval_at_k"),
            defs.get_metric_definition("unitxt.context_correctness.mrr"),
            defs.get_metric_definition("unitxt.context_correctness.map"),
        ]

        # 6. Run via RagWorkbench Experiment
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
        
        # 7. Export results
        result.export_all(
            output_path=out_dir,
            config_seq=0,
            dataset_seq=0,
            config_name=recipe_id,
            dataset_id=scenario_id,
        )

        logger.info("Experiment complete — results in %s", out_dir)
        return out_dir
