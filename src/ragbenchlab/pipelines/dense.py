import pickle
import logging
import faiss
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from typing import Any

from ragworkbench.api.ingest import IngestPipeline, IngestArtifact, IngestParams
from ragworkbench.api.inference import InferencePipeline, InferenceParams, InferenceResult

from ragbenchlab.configs.recipes import RecipeConfig
from ragbenchlab.configs.config import K_RETRIEVE, K_CONTEXT, EMBEDDING_MODEL
from ragbenchlab.utils.retrieval_utils import chunk_corpus

logger = logging.getLogger(__name__)

class DenseIngestArtifact(IngestArtifact):
    index_cache_path: str

class DenseIngestPipeline(IngestPipeline):
    def __init__(self, recipe: RecipeConfig, scenario_id: str):
        super().__init__(IngestParams())
        self.recipe = recipe
        self.scenario_id = scenario_id
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def process(self, data_loader, cache_dir: Path | str | None = None) -> list[DenseIngestArtifact]:
        if cache_dir is None:
            cache_dir = Path("cache")
        cache_dir = Path(cache_dir)
        
        index_cache_path = cache_dir / f"{self.scenario_id}_{self.recipe.recipe_id}_corpus_index.pkl"
        if index_cache_path.exists():
            logger.info(f"Using cached Dense index from {index_cache_path}")
            return [DenseIngestArtifact(index_cache_path=str(index_cache_path))]
            
        corpus_dir = cache_dir / f"corpus_files_{self.scenario_id}"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        
        corpus = data_loader.get_corpus()
        logger.info(f"Exporting corpus to {corpus_dir}")
        corpus.export_to_folder(corpus_dir)
        
        logger.info("Chunking corpus...")
        chunk_texts, chunk_ids = chunk_corpus(corpus_dir, self.tokenizer)
        
        logger.info("Encoding chunks with SentenceTransformer...")
        # L2 normalization is recommended for IndexFlatIP (cosine similarity)
        embeddings = self.model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        logger.info("Building FAISS IndexFlatIP...")
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        
        index_data = {
            "faiss_index": faiss.serialize_index(faiss_index),
            "chunk_texts": chunk_texts,
            "chunk_ids": chunk_ids,
        }
        
        # FAISS indexes aren't easily pickled with standard pickle natively,
        # but faiss.serialize_index can be used.
        # Actually, python's pickle might work for IndexFlatIP in newer FAISS versions?
        # Let's use serialize_index to be safe.
        with open(index_cache_path, "wb") as f:
            pickle.dump(index_data, f)
            
        logger.info(f"Saved Dense index to {index_cache_path}")
        return [DenseIngestArtifact(index_cache_path=str(index_cache_path))]


class DenseInferencePipeline(InferencePipeline):
    def __init__(self, recipe: RecipeConfig, scenario_id: str, params: InferenceParams | None = None, cache_dir: Path | str | None = None):
        if params is None:
            params = InferenceParams()
        super().__init__(params, cache_dir)
        self.recipe = recipe
        self.scenario_id = scenario_id
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.faiss_index = None
        self.chunk_texts = []
        self.chunk_ids = []

    def set_ingest_artifacts(self, ingest_artifacts: list[DenseIngestArtifact]):
        artifact = ingest_artifacts[0]
        
        with open(artifact.index_cache_path, "rb") as f:
            index_data = pickle.load(f)
            
        self.faiss_index = faiss.deserialize_index(index_data["faiss_index"])
        self.chunk_texts = index_data["chunk_texts"]
        self.chunk_ids = index_data["chunk_ids"]
        
    def process_no_cache(self, benchmark_entry: Any) -> InferenceResult:
        # Encode query
        query_embedding = self.model.encode([benchmark_entry.question], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        top_k = min(K_RETRIEVE, len(self.chunk_ids))
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # slices to K_CONTEXT
        context_k = min(K_CONTEXT, top_k)
        
        from ragbenchlab.utils.retrieval_utils import extract_document_ids
        
        retrieved_chunk_ids = []
        contexts = []
        
        for i in range(context_k):
            idx = indices[0][i]
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
