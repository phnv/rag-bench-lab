import pickle
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from typing import Any

from ragworkbench.api.ingest import IngestPipeline, IngestArtifact, IngestParams
from ragworkbench.api.inference import InferencePipeline, InferenceParams, InferenceResult

from ragbenchlab.configs.recipes import RecipeConfig
from ragbenchlab.configs.config import K_RETRIEVE, K_CONTEXT, EMBEDDING_MODEL
from ragbenchlab.utils.retrieval_utils import chunk_corpus

logger = logging.getLogger(__name__)

class BM25IngestArtifact(IngestArtifact):
    index_cache_path: str


class BM25IngestPipeline(IngestPipeline):
    def __init__(self, recipe: RecipeConfig, scenario_id: str):
        super().__init__(IngestParams())
        self.recipe = recipe
        self.scenario_id = scenario_id
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    def process(self, data_loader, cache_dir: Path | str | None = None) -> list[BM25IngestArtifact]:
        if cache_dir is None:
            cache_dir = Path("cache")
        cache_dir = Path(cache_dir)
        
        index_cache_path = cache_dir / f"{self.scenario_id}_{self.recipe.recipe_id}_corpus_index.pkl"
        if index_cache_path.exists():
            logger.info(f"Using cached BM25 index from {index_cache_path}")
            return [BM25IngestArtifact(index_cache_path=str(index_cache_path))]
            
        corpus_dir = cache_dir / f"corpus_files_{self.scenario_id}"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        
        corpus = data_loader.get_corpus()
        logger.info(f"Exporting corpus to {corpus_dir}")
        corpus.export_to_folder(corpus_dir)
        
        logger.info("Chunking corpus...")
        chunk_texts, chunk_ids = chunk_corpus(corpus_dir, self.tokenizer)
        
        logger.info("Tokenizing chunks for BM25...")
        tokenized_corpus = [self.tokenizer.encode(t, add_special_tokens=False) for t in chunk_texts]
        
        logger.info("Building BM25 index...")
        bm25 = BM25Okapi(tokenized_corpus)
        
        index_data = {
            "bm25_index": bm25,
            "chunk_texts": chunk_texts,
            "chunk_ids": chunk_ids,
        }
        
        with open(index_cache_path, "wb") as f:
            pickle.dump(index_data, f)
            
        logger.info(f"Saved BM25 index to {index_cache_path}")
        return [BM25IngestArtifact(index_cache_path=str(index_cache_path))]


class BM25InferencePipeline(InferencePipeline):
    def __init__(self, recipe: RecipeConfig, scenario_id: str, params: InferenceParams | None = None, cache_dir: Path | str | None = None):
        if params is None:
            params = InferenceParams()
        super().__init__(params, cache_dir)
        self.recipe = recipe
        self.scenario_id = scenario_id
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.bm25 = None
        self.chunk_texts = []
        self.chunk_ids = []

    def set_ingest_artifacts(self, ingest_artifacts: list[BM25IngestArtifact]):
        # We expect one artifact
        artifact = ingest_artifacts[0]
        
        with open(artifact.index_cache_path, "rb") as f:
            index_data = pickle.load(f)
            
        self.bm25 = index_data["bm25_index"]
        self.chunk_texts = index_data["chunk_texts"]
        self.chunk_ids = index_data["chunk_ids"]
        
    def process_no_cache(self, benchmark_entry: Any) -> InferenceResult:
        query_tokens = self.tokenizer.encode(benchmark_entry.question, add_special_tokens=False)
        
        scores = self.bm25.get_scores(query_tokens)
        
        # Sort by score descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        top_k = min(K_RETRIEVE, len(ranked))
        
        # We slice to K_CONTEXT to return in InferenceResult
        context_k = min(K_CONTEXT, top_k)
        
        from ragbenchlab.utils.retrieval_utils import extract_document_ids
        
        retrieved_chunk_ids = []
        contexts = []
        
        for i in range(context_k):
            idx = ranked[i][0]
            retrieved_chunk_ids.append(self.chunk_ids[idx])
            contexts.append(self.chunk_texts[idx])
            
        document_ids = extract_document_ids(retrieved_chunk_ids)
            
        return InferenceResult(
            question_id=benchmark_entry.question_id,
            question=benchmark_entry.question,
            answer="",
            context_ids=document_ids,
            contexts=contexts,
            ground_truth_answers=benchmark_entry.ground_truth_answers,
            ground_truths_context_ids=benchmark_entry.ground_truths_context_ids,
            is_answerable=benchmark_entry.is_answerable,
        )
