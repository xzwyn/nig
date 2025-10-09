# src/alignment/semantic_aligner.py

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

import config
from src.clients.azure_client import get_embeddings # Correctly import from Azure client

# Type Aliases
ContentItem = Dict[str, Any]
AlignedPair = Dict[str, Any]

def _get_embeddings_with_context(
    texts: List[str],
    content_items: List[ContentItem]
) -> np.ndarray:
    """
    Prepares text with context and gets embeddings from the Azure OpenAI API.
    """
    context_window = 1
    texts_with_context = []
    print(f"Enhancing text with a context window of {context_window}...")
    for i, text in enumerate(texts):
        pre_context = " ".join([content_items[j]['text'] for j in range(max(0, i - context_window), i)])
        post_context = " ".join([content_items[j]['text'] for j in range(i + 1, min(len(texts), i + 1 + context_window))])
        context_text = f"{pre_context} [SEP] {text} [SEP] {post_context}".strip()
        texts_with_context.append(context_text)
    
    # Get embeddings from Azure and convert to numpy array
    embedding_list = get_embeddings(texts_with_context)
    return np.array(embedding_list)

def _calculate_type_matrix(eng_content: List[ContentItem], ger_content: List[ContentItem]) -> np.ndarray:
    """Calculates a matrix rewarding or penalizing based on content type matching."""
    num_eng, num_ger = len(eng_content), len(ger_content)
    type_matrix = np.zeros((num_eng, num_ger))
    for i in range(num_eng):
        for j in range(num_ger):
            if eng_content[i]['type'] == ger_content[j]['type']:
                type_matrix[i, j] = config.TYPE_MATCH_BONUS
            else:
                type_matrix[i, j] = config.TYPE_MISMATCH_PENALTY
    return type_matrix

def _calculate_proximity_matrix(num_eng: int, num_ger: int) -> np.ndarray:
    """Calculates a matrix rewarding based on the relative position in the document."""
    proximity_matrix = np.zeros((num_eng, num_ger))
    for i in range(num_eng):
        for j in range(num_ger):
            norm_pos_eng = i / num_eng if num_eng > 1 else 0
            norm_pos_ger = j / num_ger if num_ger > 1 else 0
            proximity_matrix[i, j] = 1.0 - abs(norm_pos_eng - norm_pos_ger)
    return proximity_matrix

def align_content(
    english_content: List[ContentItem],
    german_content: List[ContentItem],
    generate_debug_report: bool = False
) -> Tuple[List[AlignedPair], Optional[Dict[str, Any]]]:
    """
    Aligns content using the Hungarian algorithm with Azure OpenAI embeddings.

    Returns:
        A tuple containing:
        1. A list of aligned pairs.
        2. A dictionary with all data needed for a debug report (or None if not requested).
    """
    if not english_content or not german_content:
        return [], None

    # Get embeddings for both languages
    english_embeddings = _get_embeddings_with_context(
        [item['text'] for item in english_content], english_content
    )
    german_embeddings = _get_embeddings_with_context(
        [item['text'] for item in german_content], german_content
    )

    # Calculate the three score matrices
    semantic_matrix = cosine_similarity(english_embeddings, german_embeddings)
    type_matrix = _calculate_type_matrix(english_content, german_content)
    proximity_matrix = _calculate_proximity_matrix(len(english_content), len(german_content))

    # Blend the matrices using weights from config
    blended_matrix = (
        (config.W_SEMANTIC * semantic_matrix) +
        (config.W_TYPE * type_matrix) +
        (config.W_PROXIMITY * proximity_matrix)
    )

    debug_data = None
    if generate_debug_report:
        debug_data = {
            'english_content': english_content,
            'german_content': german_content,
            'blended_matrix': blended_matrix,
            'semantic_matrix': semantic_matrix,
            'type_matrix': type_matrix,
            'proximity_matrix': proximity_matrix
        }

    # Find optimal pairs using the Hungarian algorithm
    aligned_pairs: List[AlignedPair] = []
    used_english_indices, used_german_indices = set(), set()
    cost_matrix = -blended_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    for eng_idx, ger_idx in zip(row_indices, col_indices):
        score = blended_matrix[eng_idx, ger_idx]
        if score >= config.SIMILARITY_THRESHOLD:
            semantic_score = semantic_matrix[eng_idx, ger_idx]
            aligned_pairs.append({
                "english": english_content[eng_idx],
                "german": german_content[ger_idx],
                "similarity": float(semantic_score)
            })
            used_english_indices.add(eng_idx)
            used_german_indices.add(ger_idx)

    # Add any remaining unmatched items
    for i, item in enumerate(english_content):
        if i not in used_english_indices:
            aligned_pairs.append({"english": item, "german": None, "similarity": 0.0})
    for j, item in enumerate(german_content):
        if j not in used_german_indices:
            aligned_pairs.append({"english": None, "german": item, "similarity": 0.0})
    
    return aligned_pairs, debug_data
