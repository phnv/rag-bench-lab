import pytest
from pathlib import Path
from ragbenchlab.utils.retrieval_utils import chunk_corpus, reciprocal_rank_fusion

class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()
    def decode(self, tokens):
        return " ".join(tokens)

def test_chunk_corpus(tmp_path):
    # Setup test corpus
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("a b c d e f g h i j k l m n o p q r s t") # 20 words
    
    doc2 = tmp_path / "doc2.txt"
    doc2.write_text("1 2 3") # 3 words
    
    tokenizer = MockTokenizer()
    
    # Test with chunk_size 10 and overlap 2
    chunk_texts, chunk_ids = chunk_corpus(tmp_path, tokenizer, chunk_size=10, overlap=2)
    
    # doc1:
    # 0: "a b c d e f g h i j" (10 words)
    # 1: "i j k l m n o p q r" (10 words) - overlaps "i j"
    # 2: "q r s t" (4 words) - overlaps "q r"
    # doc2:
    # 0: "1 2 3" (3 words)
    
    # Total chunks = 3 + 1 = 4
    assert len(chunk_texts) == 4
    assert len(chunk_ids) == 4
    
    assert chunk_texts[0] == "a b c d e f g h i j"
    assert chunk_ids[0] == "doc1_chunk_0"
    
    assert chunk_texts[1] == "i j k l m n o p q r"
    assert chunk_ids[1] == "doc1_chunk_1"
    
    assert chunk_texts[2] == "q r s t"
    assert chunk_ids[2] == "doc1_chunk_2"
    
    assert chunk_texts[3] == "1 2 3"
    assert chunk_ids[3] == "doc2_chunk_0"

def test_reciprocal_rank_fusion_basic():
    # List 1: A B C
    # List 2: B C D
    
    ranked_lists = [
        ["A", "B", "C"],
        ["B", "C", "D"],
    ]
    
    # k = 60
    # List 1 ranks: A=0, B=1, C=2
    # List 2 ranks: B=0, C=1, D=2
    # Scores:
    # A = 1/61
    # B = 1/62 + 1/61 = 2.0 / 61.5 approximately
    # C = 1/63 + 1/62 = 2.0 / 62.5
    # D = 1/63
    
    result = reciprocal_rank_fusion(ranked_lists, k=60)
    
    # B should be first, then C, then A, then D
    # A = 1/61 = 0.0163934426
    # B = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # C = 1/63 + 1/62 = 0.015873 + 0.016129 = 0.032002
    # D = 1/63 = 0.015873
    
    assert result == ["B", "C", "A", "D"]

def test_reciprocal_rank_fusion_determinism():
    ranked_lists = [
        ["A", "B"],
        ["C", "D"]
    ]
    
    res1 = reciprocal_rank_fusion(ranked_lists, k=60)
    res2 = reciprocal_rank_fusion(ranked_lists, k=60)
    assert res1 == res2
