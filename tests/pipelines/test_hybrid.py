import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ragbenchlab.configs.recipes import RecipeConfig, RetrievalStrategy
from ragbenchlab.pipelines.hybrid import HybridIngestPipeline, HybridInferencePipeline
from dataclasses import dataclass

@dataclass
class RagBenchmarkEntry:
    question_id: str
    question: str
    ground_truth_answers: list
    ground_truths_context_ids: list
    is_answerable: bool

class MockCorpus:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
    
    def export_to_folder(self, output_folder: Path):
        output_folder.mkdir(parents=True, exist_ok=True)
        # Using words that will be found by BM25 (keyword) and Dense (fruit context)
        (output_folder / "doc1.txt").write_text("Hello world this is a test document with some keyword")
        (output_folder / "doc2.txt").write_text("Apple fruit red round")
        (output_folder / "doc3.txt").write_text("Car engine fast drive")

class MockDataLoader:
    def __init__(self, tmp_path):
        self.corpus = MockCorpus(tmp_path)
    def get_corpus(self):
        return self.corpus

def test_hybrid_pipeline(tmp_path):
    recipe = RecipeConfig(recipe_id="R3", description="test", retrieval=RetrievalStrategy.HYBRID)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    loader = MockDataLoader(tmp_path)
    
    # Test Ingest
    ingest = HybridIngestPipeline(recipe, scenario_id="test_scenario")
    artifacts = ingest.process(loader, cache_dir)
    
    assert Path(artifacts[0].index_cache_path).exists()
    
    # Test Inference
    infer = HybridInferencePipeline(recipe, scenario_id="test_scenario")
    infer.set_ingest_artifacts(artifacts)
    
    entry = RagBenchmarkEntry(
        question_id="1",
        question="keyword about a fruit",
        ground_truth_answers=[],
        ground_truths_context_ids=[],
        is_answerable=True,
    )
    
    result = infer.process_no_cache(entry)
    
    assert len(result.context_ids) > 0
    # The hybrid result should have either doc1 or doc2 or both near the top
    assert any("doc1" in cid or "doc2" in cid for cid in result.context_ids)


def test_hybrid_reranker_pipeline(tmp_path):
    recipe = RecipeConfig(recipe_id="R4", description="test reranker", retrieval=RetrievalStrategy.HYBRID, reranker=True, fusion=True)
    cache_dir = tmp_path / "cache_r4"
    cache_dir.mkdir()
    
    loader = MockDataLoader(tmp_path)
    
    # Test Ingest (Same as R3)
    ingest = HybridIngestPipeline(recipe, scenario_id="test_scenario")
    artifacts = ingest.process(loader, cache_dir)
    
    assert Path(artifacts[0].index_cache_path).exists()
    
    # Test Inference
    infer = HybridInferencePipeline(recipe, scenario_id="test_scenario")
    infer.set_ingest_artifacts(artifacts)
    
    # Validate reranker loaded
    assert infer.reranker is not None
    
    entry = RagBenchmarkEntry(
        question_id="1",
        question="keyword about a fruit",
        ground_truth_answers=[],
        ground_truths_context_ids=[],
        is_answerable=True,
    )
    
    result = infer.process_no_cache(entry)
    
    assert len(result.context_ids) > 0
    assert any("doc1" in cid or "doc2" in cid for cid in result.context_ids)
