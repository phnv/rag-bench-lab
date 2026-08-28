import logging
from typing import Any, List
from pathlib import Path

from ragworkbench import DatasetName, Experiment, DataLoaderFactory
from ragworkbench.api.ingest import IngestPipeline, IngestArtifact, IngestParams
from ragworkbench.api.inference import InferencePipeline, InferenceResult, InferenceParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyIngestPipeline(IngestPipeline):
    def __init__(self):
        super().__init__(IngestParams())

    def process(self, data_loader: Any, **kwargs) -> IngestArtifact:
        logger.info(f"Ingesting dataset...")
        return IngestArtifact(data={"doc_count": 100})

class DummyInferencePipeline(InferencePipeline):
    def __init__(self):
        super().__init__(InferenceParams())

    def set_ingest_artifacts(self, ingest_artifacts: List[IngestArtifact]) -> None:
        pass
        
    def process_no_cache(self, benchmark_entry: Any, **kwargs) -> InferenceResult:
        return InferenceResult(
            question_id=benchmark_entry.question_id,
            question=benchmark_entry.question,
            answer="This is a dummy answer.",
            context=["dummy context chunk"],
            metadata={"source": "dummy"}
        )

def main():
    dataset_name = DatasetName.CLAP_NQ
    logger.info(f"Running smoke test with dataset: {dataset_name.name}")
    
    data_loader = DataLoaderFactory.create_loader(dataset_name)
    
    from ragworkbench.boards.board_model import ExperimentConfig
    experiment = Experiment(
        experiment_id="smoke_test",
        data_loader=data_loader,
        ingest_pipeline=DummyIngestPipeline(),
        inference_pipeline=DummyInferencePipeline(),
        eval_metrics=[], # Empty metrics for smoke test
        experiment_config=ExperimentConfig(),
        cache_dir=Path("cache")
    )
    
    # Run the experiment
    logger.info("Running experiment...")
    experiment.run()
    logger.info("Smoke test completed successfully!")

if __name__ == "__main__":
    main()
