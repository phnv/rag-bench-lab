import logging
from pathlib import Path

from ragbenchlab.configs.config import CHUNK_OVERLAP, CHUNK_SIZE, RRF_K

logger = logging.getLogger(__name__)

def chunk_corpus(
    corpus_dir: Path,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[list[str], list[str]]:
    """
    Read documents from `corpus_dir` (exported by RagCorpus), and split into token-bounded chunks.
    Returns (chunk_texts, chunk_ids) where
    chunk_id = f"{doc_id}_chunk_{i}".
    Metadata: parent doc ID attached to each chunk for Recall@K matching.
    """
    chunk_texts = []
    chunk_ids = []

    # Iterate over all text files in the corpus directory, sorted for determinism
    for filepath in sorted(corpus_dir.glob("*.txt")):
        doc_id = filepath.stem
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if not tokens:
            continue

        i = 0
        chunk_idx = 0
        while i < len(tokens):
            end = min(i + chunk_size, len(tokens))
            chunk_tokens = tokens[i:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            chunk_texts.append(chunk_text)
            chunk_ids.append(f"{doc_id}_chunk_{chunk_idx}")
            
            chunk_idx += 1
            i += chunk_size - overlap
            
            if end == len(tokens):
                break

    return chunk_texts, chunk_ids

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> list[str]:
    """
    Merge N ranked lists of chunk_ids using RRF (k=60, equal weights).
    Returns a single merged ranked list.
    Logs: rrf_k value, input list lengths, output list length.
    """
    scores = {}
    
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            if item not in scores:
                scores[item] = 0.0
            scores[item] += 1.0 / (k + rank + 1)
            
    # Sort by score descending
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    result = [item for item, score in merged]
    
    logger.info(f"RRF (k={k}) fusion: {[len(l) for l in ranked_lists]} -> {len(result)}")
    
    return result

def extract_document_ids(chunk_ids: list[str]) -> list[str]:
    """
    Extract document IDs from chunk IDs (which have the format <doc_id>_chunk_<idx>)
    and deduplicate them while preserving order.
    """
    doc_ids = []
    seen = set()
    for cid in chunk_ids:
        doc_id = cid.split("_chunk_")[0]
        if doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids
