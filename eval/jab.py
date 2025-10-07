# === Existing keys for your 4o model (if any) ===
OPENAI_API_KEY="sk-..."

# === NEW keys for Azure Embedding Model ===
AZURE_EMBEDDING_ENDPOINT="https://translation-evaluation-openai.openai.azure.com/"
AZURE_EMBEDDING_API_KEY="your-secret-api-key-from-azure"
AZURE_EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-large"
AZURE_API_VERSION="2024-02-01"


# Core libraries for Azure OpenAI and configuration
openai
azure-core
python-dotenv

# Libraries for data handling and calculations
numpy
scikit-learn
tqdm

# Libraries for dataframes and writing Excel files
pandas
openpyxl


# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Azure OpenAI Embedding Model Configuration ---
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")

# --- Document Parsing Configuration ---
# Roles to exclude from processing entirely
IGNORED_ROLES = {'pageHeader', 'pageFooter', 'pageNumber'}
# Roles that should not be stitched with other paragraphs
STRUCTURAL_ROLES = {'title', 'sectionHeading'}

# --- Alignment Algorithm Weights & Thresholds ---
# Blended score weights
W_SEMANTIC = 0.70  # Weight for semantic similarity (cosine score)
W_TYPE = 0.15      # Weight for matching content types (e.g., table vs. table)
W_PROXIMITY = 0.15 # Weight for relative position in the document

# Bonuses and Penalties
TYPE_MATCH_BONUS = 0.1
TYPE_MISMATCH_PENALTY = -0.2

# The minimum blended score for a pair to be considered a match
SIMILARITY_THRESHOLD = 0.5



# src/alignment/semantic_aligner.py

from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from tqdm import tqdm

import config
from src.reporting.excel_writer import save_calculation_report

# Type Aliases for clarity
ContentItem = Dict[str, Any]
AlignedPair = Dict[str, Any]

# A reusable client instance
_client = None

def _get_azure_client() -> AzureOpenAI:
    """Initializes and returns a reusable AzureOpenAI client."""
    global _client
    if _client is None:
        print("Initializing Azure OpenAI client...")
        if not all([config.AZURE_EMBEDDING_ENDPOINT, config.AZURE_EMBEDDING_API_KEY]):
            raise ValueError("Azure credentials (endpoint, key) are not set in the config/.env file.")
        
        _client = AzureOpenAI(
            api_version=config.AZURE_API_VERSION,
            azure_endpoint=config.AZURE_EMBEDDING_ENDPOINT,
            api_key=config.AZURE_EMBEDDING_API_KEY,
        )
    return _client

def _get_embeddings_in_batches(texts: List[str], client: AzureOpenAI, batch_size: int = 16) -> np.ndarray:
    """
    Generates embeddings by sending texts to the Azure API in batches.
    """
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=config.AZURE_EMBEDDING_DEPLOYMENT_NAME
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"An error occurred while processing a batch: {e}")
            # Add placeholder embeddings for the failed batch to avoid size mismatch
            all_embeddings.extend([[0.0] * 3072] * len(batch)) # text-embedding-3-large has 3072 dimensions

    return np.array(all_embeddings)

def _calculate_type_matrix(eng_content: List[ContentItem], ger_content: List[ContentItem]) -> np.ndarray:
    num_eng = len(eng_content)
    num_ger = len(ger_content)
    type_matrix = np.zeros((num_eng, num_ger))

    for i in range(num_eng):
        for j in range(num_ger):
            if eng_content[i]['type'] == ger_content[j]['type']:
                type_matrix[i, j] = config.TYPE_MATCH_BONUS
            else:
                type_matrix[i, j] = config.TYPE_MISMATCH_PENALTY
    return type_matrix

def _calculate_proximity_matrix(num_eng: int, num_ger: int) -> np.ndarray:
    proximity_matrix = np.zeros((num_eng, num_ger))
    for i in range(num_eng):
        for j in range(num_ger):
            norm_pos_eng = i / num_eng if num_eng > 0 else 0
            norm_pos_ger = j / num_ger if num_ger > 0 else 0
            proximity_matrix[i, j] = 1.0 - abs(norm_pos_eng - norm_pos_ger)
    return proximity_matrix

def align_content(
    english_content: List[ContentItem],
    german_content: List[ContentItem],
    generate_debug_report: bool = False,
    debug_report_path: Path = None
) -> List[AlignedPair]:
    if not english_content or not german_content:
        return []

    client = _get_azure_client()
    num_eng, num_ger = len(english_content), len(german_content)

    eng_texts = [item['text'] for item in english_content]
    ger_texts = [item['text'] for item in german_content]
    
    # Generate embeddings using the new Azure API function
    english_embeddings = _get_embeddings_in_batches(eng_texts, client)
    german_embeddings = _get_embeddings_in_batches(ger_texts, client)
    
    print("Calculating score matrices (semantic, type, proximity)...")
    semantic_matrix = cosine_similarity(english_embeddings, german_embeddings)
    type_matrix = _calculate_type_matrix(english_content, german_content)
    proximity_matrix = _calculate_proximity_matrix(num_eng, num_ger)

    blended_matrix = (
        (config.W_SEMANTIC * semantic_matrix) +
        (config.W_TYPE * type_matrix) +
        (config.W_PROXIMITY * proximity_matrix)
    )

    if generate_debug_report and debug_report_path:
        print("Generating detailed calculation report for debugging...")
        save_calculation_report(
            english_content=english_content,
            german_content=german_content,
            blended_matrix=blended_matrix,
            semantic_matrix=semantic_matrix,
            type_matrix=type_matrix,
            proximity_matrix=proximity_matrix,
            filepath=debug_report_path
        )

    print("Finding best matches based on blended scores...")
    aligned_pairs: List[AlignedPair] = []
    used_german_indices = set()
    
    best_ger_matches = np.argmax(blended_matrix, axis=1)
    best_eng_matches = np.argmax(blended_matrix, axis=0)

    for eng_idx, ger_idx in enumerate(best_ger_matches):
        is_mutual_best_match = (best_eng_matches[ger_idx] == eng_idx)
        score = blended_matrix[eng_idx, ger_idx]

        if is_mutual_best_match and score >= config.SIMILARITY_THRESHOLD:
            semantic_score = semantic_matrix[eng_idx, ger_idx]
            aligned_pairs.append({
                "english": english_content[eng_idx],
                "german": german_content[ger_idx],
                "similarity": float(semantic_score) # Cast to float for JSON serialization
            })
            used_german_indices.add(ger_idx)

    matched_english_ids = {id(pair['english']) for pair in aligned_pairs if pair.get('english')}
    for item in english_content:
        if id(item) not in matched_english_ids:
            aligned_pairs.append({"english": item, "german": None, "similarity": 0.0})

    for idx, item in enumerate(german_content):
        if idx not in used_german_indices:
             aligned_pairs.append({"english": None, "german": item, "similarity": 0.0})

    aligned_pairs.sort(key=lambda x: x['english']['page'] if x.get('english') else float('inf'))
    
    return aligned_pairs
