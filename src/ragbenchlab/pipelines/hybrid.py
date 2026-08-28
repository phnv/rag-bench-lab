import pickle
import logging
import faiss
import numpy as np
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from typing import Any

from ragworkbench.api.ingest import IngestPipeline, IngestArtifact, IngestParams
from ragworkbench.api.inference import InferencePipeline, InferenceParams, InferenceResult

from ragbenchlab.configs.recipes import RecipeConfig
from ragbenchlab.configs.config import K_RETRIEVE, K_CONTEXT, EMBEDDING_MODEL, RRF_K, RERANKER_MODEL
from ragbenchlab.utils.retrieval_utils import chunk_corpus, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

class HybridIngestArtifact(IngestArtifact):
    index_cache_path: str


class HybridIngestPipeline(IngestPipeline):
    def __init__(self, recipe: RecipeConfig, scenario_id: str):
        super().__init__(IngestParams())
        self.recipe = recipe
        self.scenario_id = scenario_id
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def process(self, data_loader, cache_dir: Path | str | None = None) -> list[HybridIngestArtifact]:
        if cache_dir is None:
            cache_dir = Path("cache")
        cache_dir = Path(cache_dir)
        
        index_cache_path = cache_dir / f"{self.scenario_id}_{self.recipe.recipe_id}_corpus_index.pkl"
        if index_cache_path.exists():
            logger.info(f"Using cached Hybrid index from {index_cache_path}")
            return [HybridIngestArtifact(index_cache_path=str(index_cache_path))]

        # Check for R1 and R2 caches to reuse
        r1_cache_path = cache_dir / f"{self.scenario_id}_R1_corpus_index.pkl"
        r2_cache_path = cache_dir / f"{self.scenario_id}_R2_corpus_index.pkl"

        if r1_cache_path.exists() and r2_cache_path.exists():
            logger.info("Found R1 and R2 cached indexes. Attempting to merge...")
            try:
                with open(r1_cache_path, "rb") as f:
                    r1_data = pickle.load(f)
                with open(r2_cache_path, "rb") as f:
                    r2_data = pickle.load(f)

                if r1_data["chunk_ids"] == r2_data["chunk_ids"]:
                    logger.info("Chunk IDs match. Merging R1 (BM25) and R2 (FAISS) caches into R3 cache.")
                    
                    index_data = {
                        "bm25_index": r1_data["bm25_index"],
                        "faiss_index": r2_data["faiss_index"],
                        "chunk_texts": r1_data["chunk_texts"],
                        "chunk_ids": r1_data["chunk_ids"],
                    }
                    
                    with open(index_cache_path, "wb") as f:
                        pickle.dump(index_data, f)
                        
                    logger.info(f"Saved merged Hybrid index to {index_cache_path}")
                    return [HybridIngestArtifact(index_cache_path=str(index_cache_path))]
                else:
                    logger.warning("R1 and R2 cache chunk_ids do not match! Configuration drift detected. Falling back to from-scratch generation.")
            except Exception as e:
                logger.warning(f"Failed to merge R1 and R2 caches: {e}. Falling back to from-scratch generation.")
        else:
            logger.info("R1 and/or R2 cached indexes not found. Falling back to from-scratch generation.")
            
        corpus_dir = cache_dir / f"corpus_files_{self.scenario_id}"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        
        corpus = data_loader.get_corpus()
        logger.info(f"Exporting corpus to {corpus_dir}")
        corpus.export_to_folder(corpus_dir)
        
        logger.info("Chunking corpus...")
        chunk_texts, chunk_ids = chunk_corpus(corpus_dir, self.tokenizer)
        
        logger.info("Building BM25 index...")
        tokenized_corpus = [self.tokenizer.encode(t, add_special_tokens=False) for t in chunk_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("Building FAISS IndexFlatIP...")
        embeddings = self.model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        
        index_data = {
            "bm25_index": bm25,
            "faiss_index": faiss.serialize_index(faiss_index),
            "chunk_texts": chunk_texts,
            "chunk_ids": chunk_ids,
        }
        
        with open(index_cache_path, "wb") as f:
            pickle.dump(index_data, f)
            
        logger.info(f"Saved Hybrid index to {index_cache_path}")
        return [HybridIngestArtifact(index_cache_path=str(index_cache_path))]


class HybridInferencePipeline(InferencePipeline):
    def __init__(self, recipe: RecipeConfig, scenario_id: str, params: InferenceParams | None = None, cache_dir: Path | str | None = None):
        if params is None:
            params = InferenceParams()
        super().__init__(params, cache_dir)
        self.recipe = recipe
        self.scenario_id = scenario_id
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
        self.reranker = None
        if self.recipe.reranker:
            logger.info(f"Loading CrossEncoder: {RERANKER_MODEL}")
            self.reranker = CrossEncoder(RERANKER_MODEL)
        
        self.bm25 = None
        self.faiss_index = None
        self.chunk_texts = []
        self.chunk_ids = []
        self.chunk_id_to_text = {}

    def set_ingest_artifacts(self, ingest_artifacts: list[HybridIngestArtifact]):
        artifact = ingest_artifacts[0]
        
        with open(artifact.index_cache_path, "rb") as f:
            index_data = pickle.load(f)
            
        self.bm25 = index_data["bm25_index"]
        self.faiss_index = faiss.deserialize_index(index_data["faiss_index"])
        self.chunk_texts = index_data["chunk_texts"]
        self.chunk_ids = index_data["chunk_ids"]
        
        self.chunk_id_to_text = dict(zip(self.chunk_ids, self.chunk_texts))
        
    def process_no_cache(self, benchmark_entry: Any) -> InferenceResult:
        # BM25 Retrieval
        query_tokens = self.tokenizer.encode(benchmark_entry.question, add_special_tokens=False)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_top_k = min(K_RETRIEVE, len(bm25_ranked))
        bm25_candidates = [self.chunk_ids[bm25_ranked[i][0]] for i in range(bm25_top_k)]
        
        # Dense Retrieval
        query_embedding = self.model.encode([benchmark_entry.question], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        dense_top_k = min(K_RETRIEVE, len(self.chunk_ids))
        _, indices = self.faiss_index.search(query_embedding, dense_top_k)
        dense_candidates = [self.chunk_ids[indices[0][i]] for i in range(dense_top_k)]
        
        # RRF Fusion
        fused_candidates = reciprocal_rank_fusion([bm25_candidates, dense_candidates], k=RRF_K)
        
        if self.reranker is not None:
            # Bound reranker to top K_RETRIEVE
            candidates_to_rerank = fused_candidates[:K_RETRIEVE]
            pairs = [[benchmark_entry.question, self.chunk_id_to_text[cid]] for cid in candidates_to_rerank]
            
            rerank_scores = self.reranker.predict(pairs)
            
            # Sort by reranker score descending
            ranked = sorted(enumerate(rerank_scores), key=lambda x: x[1], reverse=True)
            reranked_candidates = [candidates_to_rerank[i] for i, _ in ranked]
            
            # Slice to K_CONTEXT
            context_k = min(K_CONTEXT, len(reranked_candidates))
            retrieved_chunk_ids = reranked_candidates[:context_k]
        else:
            # Slice to K_CONTEXT
            context_k = min(K_CONTEXT, len(fused_candidates))
            retrieved_chunk_ids = fused_candidates[:context_k]
            
        from ragbenchlab.utils.retrieval_utils import extract_document_ids
            
        contexts = [self.chunk_id_to_text[cid] for cid in retrieved_chunk_ids]
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
