# src/alignment/toc_aligner.py

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from src.clients.azure_client import get_embeddings # Import the new function

# Type Aliases for clarity
ToCItem = Dict[str, Any]
AlignedToCPair = Dict[str, Any]

def align_tocs(english_toc: List[ToCItem], german_toc: List[ToCItem]) -> List[AlignedToCPair]:
    """
    Aligns the Table of Contents using Azure OpenAI embeddings.
    """
    if not english_toc or not german_toc:
        return []

    eng_titles = [item['title'] for item in english_toc]
    ger_titles = [item['title'] for item in german_toc]
    
    # Get embeddings from Azure OpenAI API
    english_embeddings_list = get_embeddings(eng_titles)
    german_embeddings_list = get_embeddings(ger_titles)

    # Convert to numpy arrays for calculation
    english_embeddings = np.array(english_embeddings_list)
    german_embeddings = np.array(german_embeddings_list)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(english_embeddings, german_embeddings)
    
    # Use the Hungarian algorithm to find the optimal assignment
    cost_matrix = -similarity_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    aligned_sections: List[AlignedToCPair] = []
    
    print("Matching ToC sections...")
    for eng_idx, ger_idx in zip(row_indices, col_indices):
        score = similarity_matrix[eng_idx, ger_idx]
        if score > 0.4:
            aligned_sections.append({
                'english': english_toc[eng_idx],
                'german': german_toc[ger_idx],
                'similarity': score
            })
            print(f"  - Matched '{english_toc[eng_idx]['title']}' -> '{german_toc[ger_idx]['title']}' (Score: {score:.2f})")

    return aligned_sections