#!/usr/bin/env python3
"""
FINAL REFINED: State-of-the-Art Bilingual Document Alignment Pipeline

- Uses high-resolution PDF layout partitioning with unstructured.io
- Chunks text using unstructured's sophisticated chunk_by_title for fine-grained, paragraph-level segmentation.
- Uses a hybrid scoring model (semantic + positional + structural)
- Performs Needleman-Wunsch global sequence alignment to find the optimal match
"""

import logging
from pathlib import Path
from typing import List, Any, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
# Import the specific partitioner and the NEW title-based chunker
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from nltk import sent_tokenize
import nltk

# --- Configuration ---
BASE_DIR = Path("C:/Users/sripa/OneDrive/Documents/alignment")
PDF_PAIRS = [
    {
        "english": BASE_DIR / "67.pdf",
        "german": BASE_DIR / "67g.pdf",
        "output_excel": BASE_DIR / "alignment_output_stateofart_final_refined.xlsx"
    }
]

# Note: 'large' models are powerful but require more VRAM.
# For GPUs with <= 4GB VRAM, 'intfloat/multilingual-e5-small' is the recommended choice.
EMBEDDING_MODEL = "intfloat/multilingual-e5-large" 
SIMILARITY_THRESHOLD = 0.70  # Threshold to flag a potential mismatch for review
POSITION_WEIGHT = 0.2         # How much to value positional similarity (0.0 to 1.0)
STRUCTURAL_WEIGHT = 0.1       # How much to value structural similarity
STRUCTURAL_MATCH_SCORE = 1.0
STRUCTURAL_MISMATCH_PENALTY = -1.0
GAP_PENALTY = -0.5

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("StateOfArtAlignment")

# Download NLTK tokenizer models if not present
try:
    sent_tokenize("test sentence.")
except LookupError:
    nltk.download('punkt')


def extract_elements(pdf_path: Path) -> List[Element]:
    """
    Partition PDF into a list of Element objects using the 'hi_res' strategy.
    Filter out headers, footers, and other noise elements.
    """
    logger.info(f"Partitioning PDF with 'hi_res' strategy: {pdf_path.name}")
    # This function returns a list of unstructured.io Element objects
    elements = partition_pdf(filename=str(pdf_path), strategy="hi_res", infer_table_structure=True)
    
    # Filter out noise and return the list of clean Element objects
    filtered_elements = [
        el for el in elements
        if el.category not in {"Header", "Footer", "PageBreak", "PageNumber"}
        and el.text and len(el.text.strip()) >= 10
    ]
    
    logger.info(f"Extracted and filtered {len(filtered_elements)} structural elements from {pdf_path.name}")
    return filtered_elements


def embed_chunks(model: SentenceTransformer, chunks: List[dict]) -> np.ndarray:
    """Encodes text chunks into normalized embeddings."""
    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} text chunks...")
    emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    # Normalize embeddings to unit length for cosine similarity
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def compute_hybrid_score(
    i: int, j: int,
    eng_chunks: List[dict], ger_chunks: List[dict],
    eng_emb: np.ndarray, ger_emb: np.ndarray,
) -> float:
    """Computes a weighted score combining semantic, positional, and structural similarity."""
    semantic_score = float(np.dot(eng_emb[i], ger_emb[j]))
    pos_e = i / (len(eng_chunks) - 1 if len(eng_chunks) > 1 else 1)
    pos_g = j / (len(ger_chunks) - 1 if len(ger_chunks) > 1 else 1)
    positional_score = 1.0 - abs(pos_e - pos_g)
    type_e, type_g = eng_chunks[i]["type"], ger_chunks[j]["type"]
    structural_score = STRUCTURAL_MATCH_SCORE if type_e == type_g else STRUCTURAL_MISMATCH_PENALTY
    
    final_score = (
        (1 - POSITION_WEIGHT - STRUCTURAL_WEIGHT) * semantic_score +
        POSITION_WEIGHT * positional_score +
        STRUCTURAL_WEIGHT * structural_score
    )
    return final_score


def needleman_wunsch(score_matrix: np.ndarray, gap_penalty: float) -> List[Tuple[int, int]]:
    """Performs global sequence alignment to find the optimal path."""
    m, n = score_matrix.shape
    dp = np.zeros((m + 1, n + 1))
    ptr = np.zeros((m + 1, n + 1), dtype=int)  # 0=diag, 1=up, 2=left

    for i in range(1, m + 1): dp[i, 0] = i * gap_penalty; ptr[i, 0] = 1
    for j in range(1, n + 1): dp[0, j] = j * gap_penalty; ptr[0, j] = 2

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1, j - 1] + score_matrix[i - 1, j - 1]
            delete = dp[i - 1, j] + gap_penalty
            insert = dp[i, j - 1] + gap_penalty
            scores = [match, delete, insert]
            best_score = max(scores)
            dp[i, j] = best_score
            ptr[i, j] = scores.index(best_score)

    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        direction = ptr[i, j]
        if direction == 0:
            alignment.append((i - 1, j - 1)); i -= 1; j -= 1
        elif direction == 1:
            alignment.append((i - 1, -1)); i -= 1
        else:
            alignment.append((-1, j - 1)); j -= 1
            
    return alignment[::-1]


def align_pair(eng_pdf: Path, ger_pdf: Path, output_xlsx: Path) -> None:
    """Main alignment logic for a single pair of documents."""
    eng_elements = extract_elements(eng_pdf)
    ger_elements = extract_elements(ger_pdf)

    # --- REFINED CHUNKING LOGIC ---
    # Replace the manual chunker with the more powerful and configurable official function.
    # This preserves paragraph boundaries within sections for finer granularity.
    logger.info("Chunking documents using unstructured's chunk_by_title...")
    eng_chunk_objects = chunk_by_title(
        eng_elements,
        max_characters=1500,
        combine_text_under_n_chars=250 # Prevents merging of distinct, reasonably-sized paragraphs
    )
    ger_chunk_objects = chunk_by_title(
        ger_elements,
        max_characters=1500,
        combine_text_under_n_chars=250
    )

    # Convert the returned Chunk objects to the dictionary format expected by downstream functions
    eng_chunks = [{"text": c.text, "type": c.metadata.category if hasattr(c.metadata, 'category') else 'Chunk'} for c in eng_chunk_objects]
    ger_chunks = [{"text": c.text, "type": c.metadata.category if hasattr(c.metadata, 'category') else 'Chunk'} for c in ger_chunk_objects]
    
    logger.info(f"Final chunk count: {len(eng_chunks)} English, {len(ger_chunks)} German")

    if not eng_chunks or not ger_chunks:
        logger.error("No chunks were created from one or both documents; cannot align.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' onto '{device}' device.")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    eng_emb = embed_chunks(model, eng_chunks)
    ger_emb = embed_chunks(model, ger_chunks)

    logger.info("Computing hybrid score matrix...")
    score_matrix = np.zeros((len(eng_chunks), len(ger_chunks)), dtype=float)
    for i in range(len(eng_chunks)):
        for j in range(len(ger_chunks)):
            score_matrix[i, j] = compute_hybrid_score(i, j, eng_chunks, ger_chunks, eng_emb, ger_emb)

    logger.info("Performing Needleman-Wunsch global alignment...")
    aligned_indices = needleman_wunsch(score_matrix, GAP_PENALTY)

    rows = []
    for i, j in aligned_indices:
        eng_text = eng_chunks[i]["text"] if i != -1 else "--- GAP ---"
        ger_text = ger_chunks[j]["text"] if j != -1 else "--- GAP ---"
        score = score_matrix[i, j] if i != -1 and j != -1 else GAP_PENALTY
        
        rows.append({
            "English_Chunk": eng_text,
            "German_Chunk": ger_text,
            "Hybrid_Score": round(score, 4),
            "Needs_Review": bool(score < SIMILARITY_THRESHOLD)
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_xlsx, index=False, engine="openpyxl")
    logger.info(f"Alignment complete. Results saved to {output_xlsx}")


def main():
    try:
        for pair in PDF_PAIRS:
            logger.info(f"--- Starting alignment for {pair['english'].name} ↔ {pair['german'].name} ---")
            align_pair(pair["english"], pair["german"], pair["output_excel"])
            logger.info(f"✅ Successfully finished alignment for the pair.")
    except Exception as e:
        logger.critical("A critical error occurred in the pipeline.", exc_info=True)
        exit(1)


if _name_ == "_main_":
    main()
