"""
Unit tests for src/ragbenchlab/runner/experiment_runner.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragbenchlab.configs.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    K_CONTEXT,
    K_RETRIEVE,
    RERANKER_MODEL,
    RRF_K,
    get_scenario,
)
from ragbenchlab.configs.recipes import get_recipe
from ragbenchlab.runner.experiment_runner import (
    ExperimentRunner,
    build_experiment_metadata,
    resolve_models_for_recipe,
)


class TestResolveModelsForRecipe:
    def test_r0_no_rag_has_no_local_models(self):
        recipe = get_recipe("R0")
        models_by_role, active_models = resolve_models_for_recipe(recipe)
        assert models_by_role == {
            "tokenizer": None,
            "embedding_model": None,
            "reranker_model": None,
            "generation_model": None,
        }
        assert active_models == []

    def test_r1_bm25_uses_tokenizer_only(self):
        recipe = get_recipe("R1")
        models_by_role, active_models = resolve_models_for_recipe(recipe)
        assert models_by_role == {
            "tokenizer": EMBEDDING_MODEL,
            "embedding_model": None,
            "reranker_model": None,
            "generation_model": None,
        }
        assert active_models == [EMBEDDING_MODEL]

    @pytest.mark.parametrize("recipe_id", ["R2", "R3"])
    def test_r2_and_r3_use_tokenizer_and_embedding_model(self, recipe_id):
        recipe = get_recipe(recipe_id)
        models_by_role, active_models = resolve_models_for_recipe(recipe)
        assert models_by_role == {
            "tokenizer": EMBEDDING_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": None,
            "generation_model": None,
        }
        assert active_models == [EMBEDDING_MODEL]

    def test_r4_uses_embedding_and_reranker_models(self):
        recipe = get_recipe("R4")
        models_by_role, active_models = resolve_models_for_recipe(recipe)
        assert models_by_role == {
            "tokenizer": EMBEDDING_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "generation_model": None,
        }
        assert active_models == [EMBEDDING_MODEL, RERANKER_MODEL]


class TestBuildExperimentMetadata:
    def test_metadata_schema_completeness(self):
        scenario = get_scenario("nq")
        recipe = get_recipe("R4")
        meta = build_experiment_metadata(
            experiment_id="nq__R4__eval_n10_s42",
            timestamp_utc="2026-11-04T20:00:00Z",
            mode="eval",
            scenario=scenario,
            recipe=recipe,
            question_limit=10,
            description="Test run description",
            comments=["first comment", "second comment"],
        )

        assert meta["experiment_id"] == "nq__R4__eval_n10_s42"
        assert meta["timestamp_utc"] == "2026-11-04T20:00:00Z"
        assert meta["mode"] == "eval"
        assert meta["description"] == "Test run description"
        assert meta["comments"] == ["first comment", "second comment"]

        assert meta["scenario"] == {
            "scenario_id": "nq",
            "dataset_name": scenario.dataset_name.value,
            "question_limit": 10,
            "seed": 42,
            "description": scenario.description,
        }

        assert meta["recipe"] == {
            "recipe_id": "R4",
            "description": recipe.description,
            "retrieval": "hybrid",
            "reranker": True,
            "top_k": 10,
            "query_transform": "original",
            "fusion": True,
            "context_strategy": "top_k",
        }

        assert meta["retrieval_params"] == {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "k_retrieve": K_RETRIEVE,
            "k_context": K_CONTEXT,
            "rrf_k": RRF_K,
        }

        assert meta["models"] == {
            "tokenizer": EMBEDDING_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "generation_model": None,
        }


class TestExperimentRunnerRun:
    @patch("ragbenchlab.runner.experiment_runner.Experiment")
    @patch("ragbenchlab.runner.experiment_runner.load_metric_definitions")
    @patch("ragbenchlab.runner.experiment_runner.build_pipeline")
    @patch("ragbenchlab.runner.experiment_runner.DataLoaderFactory.create_loader")
    def test_runner_creates_timestamped_subfolder_and_writes_metadata(
        self,
        mock_create_loader,
        mock_build_pipeline,
        mock_load_metrics,
        mock_experiment_cls,
        tmp_path: Path,
    ):
        mock_create_loader.return_value = MagicMock()
        mock_build_pipeline.return_value = (MagicMock(), MagicMock())
        mock_defs = MagicMock()
        mock_load_metrics.return_value = mock_defs

        mock_result = MagicMock()
        mock_result.cost_data.models_used = []
        mock_experiment_cls.return_value.run.return_value = mock_result

        runner = ExperimentRunner(
            output_root=tmp_path / "output",
            cache_root=tmp_path / "cache",
        )

        out_dir = runner.run(
            recipe_id="R4",
            scenario_id="nq",
            mode="eval",
            description="Unit test run",
            comments=["c1", "c2"],
        )

        # Verify subfolder naming pattern: eval_n10_s42_YYYYMMDD_HHMMSS
        assert out_dir.parent == tmp_path / "output" / "nq" / "R4"
        assert re.match(r"^eval_n\d+_s42_\d{8}_\d{6}$", out_dir.name)

        scenario = get_scenario("nq")
        expected_exp_id = f"nq__R4__eval_n{scenario.eval_size}_s{scenario.seed}"

        # Verify local models were populated in cost_data.models_used
        assert mock_result.cost_data.models_used == [
            EMBEDDING_MODEL,
            RERANKER_MODEL,
        ]

        # Verify export_all was called on the subfolder
        mock_result.export_all.assert_called_once_with(
            output_path=out_dir,
            config_seq=0,
            dataset_seq=0,
            config_name="R4",
            dataset_id="nq",
        )

        # Verify 3rd metadata file exists and has the right content
        meta_file = out_dir / f"experiment_metadata_{expected_exp_id}.json"
        assert meta_file.exists()

        with open(meta_file) as f:
            meta = json.load(f)

        assert meta["experiment_id"] == expected_exp_id
        assert meta["description"] == "Unit test run"
        assert meta["comments"] == ["c1", "c2"]
        assert meta["models"]["reranker_model"] == RERANKER_MODEL

