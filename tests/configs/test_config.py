"""
Unit tests for src/ragbenchlab/configs/config.py and recipes.py.

These tests exercise the config / recipe layer in isolation — no network,
no RagWorkbench dataset downloads.
"""

from __future__ import annotations

import pytest
from ragworkbench import DatasetName

from ragbenchlab.configs.config import (
    DEV_SIZE,
    EVAL_SIZE,
    SCENARIOS,
    SEED,
    ScenarioConfig,
    get_scenario,
)
from ragbenchlab.configs.recipes import (
    RECIPES,
    ContextStrategy,
    QueryTransform,
    RecipeConfig,
    RetrievalStrategy,
    get_recipe,
)


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------


def test_seed_is_deterministic():
    assert SEED == 42


def test_dev_size_leq_eval_size():
    assert DEV_SIZE <= EVAL_SIZE, "DEV_SIZE must not exceed EVAL_SIZE (dev ⊂ eval)"


# ---------------------------------------------------------------------------
# ScenarioConfig — structure
# ---------------------------------------------------------------------------


class TestScenarioConfig:
    def test_four_scenarios_defined(self):
        assert len(SCENARIOS) == 4

    def test_all_scenario_ids_present(self):
        expected = {"nq", "hotpotqa", "watsonx", "dabstep"}
        assert set(SCENARIOS.keys()) == expected

    @pytest.mark.parametrize("scenario_id", ["nq", "hotpotqa", "watsonx", "dabstep"])
    def test_scenario_has_correct_seed(self, scenario_id):
        sc = get_scenario(scenario_id)
        assert sc.seed == SEED

    @pytest.mark.parametrize("scenario_id", ["nq", "hotpotqa", "watsonx", "dabstep"])
    def test_scenario_has_correct_sizes(self, scenario_id):
        sc = get_scenario(scenario_id)
        assert sc.dev_size == DEV_SIZE
        assert sc.eval_size == EVAL_SIZE

    def test_nq_uses_clap_nq(self):
        assert get_scenario("nq").dataset_name == DatasetName.CLAP_NQ

    def test_hotpotqa_uses_hotpot_qa(self):
        assert get_scenario("hotpotqa").dataset_name == DatasetName.HOTPOT_QA

    def test_watsonx_uses_txt_variant(self):
        assert get_scenario("watsonx").dataset_name == DatasetName.WATSONX_DOCS_QA_TXT

    def test_dabstep_uses_dabstep(self):
        assert get_scenario("dabstep").dataset_name == DatasetName.DABSTEP

    def test_scenario_config_is_frozen(self):
        sc = get_scenario("nq")
        with pytest.raises((AttributeError, TypeError)):
            sc.seed = 99  # type: ignore[misc]

    def test_get_scenario_raises_on_unknown(self):
        with pytest.raises(KeyError, match="Unknown scenario_id"):
            get_scenario("nonexistent")


# ---------------------------------------------------------------------------
# RecipeConfig — Phase 1 baseline ladder
# ---------------------------------------------------------------------------


class TestRecipeConfig:
    def test_five_phase1_recipes_defined(self):
        # R0–R4
        assert len(RECIPES) >= 5

    @pytest.mark.parametrize("recipe_id", ["R0", "R1", "R2", "R3", "R4"])
    def test_all_baseline_recipes_present(self, recipe_id):
        assert recipe_id in RECIPES

    def test_r0_is_no_rag(self):
        r = get_recipe("R0")
        assert r.retrieval == RetrievalStrategy.NONE
        assert not r.reranker
        assert not r.fusion

    def test_r1_is_bm25(self):
        r = get_recipe("R1")
        assert r.retrieval == RetrievalStrategy.BM25
        assert not r.reranker
        assert not r.fusion

    def test_r2_is_dense(self):
        r = get_recipe("R2")
        assert r.retrieval == RetrievalStrategy.DENSE
        assert not r.reranker
        assert not r.fusion

    def test_r3_is_hybrid_with_fusion(self):
        r = get_recipe("R3")
        assert r.retrieval == RetrievalStrategy.HYBRID
        assert r.fusion
        assert not r.reranker

    def test_r4_is_hybrid_with_reranker(self):
        r = get_recipe("R4")
        assert r.retrieval == RetrievalStrategy.HYBRID
        assert r.fusion
        assert r.reranker

    @pytest.mark.parametrize("recipe_id", ["R0", "R1", "R2", "R3", "R4"])
    def test_default_query_transform_is_original(self, recipe_id):
        r = get_recipe(recipe_id)
        assert r.query_transform == QueryTransform.ORIGINAL

    @pytest.mark.parametrize("recipe_id", ["R0", "R1", "R2", "R3", "R4"])
    def test_default_context_strategy_is_top_k(self, recipe_id):
        r = get_recipe(recipe_id)
        assert r.context_strategy == ContextStrategy.TOP_K

    @pytest.mark.parametrize("recipe_id", ["R0", "R1", "R2", "R3", "R4"])
    def test_recipe_ids_match_dict_keys(self, recipe_id):
        """recipe_id field must match the dict key."""
        r = get_recipe(recipe_id)
        assert r.recipe_id == recipe_id

    def test_recipe_config_is_frozen(self):
        r = get_recipe("R0")
        with pytest.raises((AttributeError, TypeError)):
            r.top_k = 999  # type: ignore[misc]

    def test_get_recipe_raises_on_unknown(self):
        with pytest.raises(KeyError, match="Unknown recipe_id"):
            get_recipe("Z99")
