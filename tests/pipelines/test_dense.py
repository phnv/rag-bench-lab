import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ragbenchlab.configs.recipes import RecipeConfig, RetrievalStrategy
from ragbenchlab.pipelines.dense import DenseIngestPipeline, DenseInferencePipeline
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
        (output_folder / "doc1.txt").write_text("Apple fruit red round")
        (output_folder / "doc2.txt").write_text("Car engine fast drive")

class MockDataLoader:
    def __init__(self, tmp_path):
        self.corpus = MockCorpus(tmp_path)
    def get_corpus(self):
        return self.corpus

def test_dense_pipeline(tmp_path):
    recipe = RecipeConfig(recipe_id="R2", description="test", retrieval=RetrievalStrategy.DENSE)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    loader = MockDataLoader(tmp_path)
    
    # Test Ingest
    ingest = DenseIngestPipeline(recipe, scenario_id="test_scenario")
    artifacts = ingest.process(loader, cache_dir)
    
    assert Path(artifacts[0].index_cache_path).exists()
    
    # Test Inference
    infer = DenseInferencePipeline(recipe, scenario_id="test_scenario")
    infer.set_ingest_artifacts(artifacts)
    
    entry = RagBenchmarkEntry(
        question_id="1",
        question="A fruit that is red",
        ground_truth_answers=[],
        ground_truths_context_ids=[],
        is_answerable=True,
    )
    
    result = infer.process_no_cache(entry)
    
    assert len(result.context_ids) > 0
    # Top 1 chunk should be from doc1 because query is about fruit
    assert "doc1" in result.context_ids[0]
