// src/alignment/semantic_aligner.py
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment  # New import for Hungarian algorithm

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

def _get_embeddings_in_batches(
    texts: List[str], 
    content_items: List[ContentItem],  # New parameter
    client: AzureOpenAI, 
    batch_size: int = 16,
    context_window: int = 0  # New parameter
) -> np.ndarray:
    """
    Generates embeddings by sending texts to the Azure API in batches.
    Optionally includes context from surrounding segments.
    """
    # Generate texts with context if context_window > 0
    if context_window > 0:
        texts_with_context = []
        for i, text in enumerate(texts):
            # Get preceding context
            pre_context = ""
            for j in range(max(0, i - context_window), i):
                pre_context += f"{content_items[j]['text']} "

            # Get following context
            post_context = ""
            for j in range(i + 1, min(len(texts), i + context_window + 1)):
                post_context += f" {content_items[j]['text']}"

            # Include content type and page number for additional context
            content_type = content_items[i]['type']
            page_num = content_items[i]['page']

            # Create context-enhanced text
            if pre_context or post_context:
                context_text = f"{pre_context}[SEP]{text}[SEP]{post_context} [TYPE:{content_type}] [PAGE:{page_num}]"
            else:
                context_text = f"{text} [TYPE:{content_type}] [PAGE:{page_num}]"

            texts_with_context.append(context_text)

        # Use the context-enhanced texts
        texts_to_embed = texts_with_context
    else:
        # Use original texts
        texts_to_embed = texts

    # Generate embeddings in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Generating Embeddings"):
        batch = texts_to_embed[i:i + batch_size]
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
            all_embeddings.extend([[0.0] * 3072] * len(batch))  # text-embedding-3-large has 3072 dimensions

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
    algorithm: str = "mutual",  # New parameter
    context_window: int = 0,    # New parameter
    generate_debug_report: bool = False,
    debug_report_path: Path = None
) -> List[AlignedPair]:
    """
    Aligns content between English and German documents.

    Args:
        english_content: List of English content items
        german_content: List of German content items
        algorithm: Matching algorithm to use ("mutual" or "hungarian")
        context_window: Size of context window (0 for no context)
        generate_debug_report: Whether to generate a detailed calculation report
        debug_report_path: Path to save the debug report

    Returns:
        List of aligned pairs
    """
    if not english_content or not german_content:
        return []

    client = _get_azure_client()
    num_eng, num_ger = len(english_content), len(german_content)

    eng_texts = [item['text'] for item in english_content]
    ger_texts = [item['text'] for item in german_content]

    # Generate embeddings using the updated function with context support
    english_embeddings = _get_embeddings_in_batches(
        eng_texts, 
        english_content,  # Pass content items for context
        client,
        context_window=context_window
    )
    german_embeddings = _get_embeddings_in_batches(
        ger_texts, 
        german_content,  # Pass content items for context
        client,
        context_window=context_window
    )

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

    aligned_pairs: List[AlignedPair] = []
    used_german_indices = set()

    # Choose matching algorithm based on parameter
    if algorithm == "hungarian":
        print("Finding optimal global alignment using Hungarian algorithm...")
        # Hungarian algorithm minimizes cost, so we negate the similarity scores
        cost_matrix = -blended_matrix.copy()

        # Find optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Process the matches found by the Hungarian algorithm
        for eng_idx, ger_idx in zip(row_indices, col_indices):
            score = blended_matrix[eng_idx, ger_idx]

            # Only include matches that meet the threshold
            if score >= config.SIMILARITY_THRESHOLD:
                semantic_score = semantic_matrix[eng_idx, ger_idx]
                aligned_pairs.append({
                    "english": english_content[eng_idx],
                    "german": german_content[ger_idx],
                    "similarity": float(semantic_score)  # Cast to float for JSON serialization
                })
                used_german_indices.add(ger_idx)
    else:
        print("Finding best matches based on mutual best match algorithm...")
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
                    "similarity": float(semantic_score)  # Cast to float for JSON serialization
                })
                used_german_indices.add(ger_idx)

    # Add unmatched English content
    matched_english_ids = {id(pair['english']) for pair in aligned_pairs if pair.get('english')}
    for item in english_content:
        if id(item) not in matched_english_ids:
            aligned_pairs.append({"english": item, "german": None, "similarity": 0.0})

    # Add unmatched German content
    for idx, item in enumerate(german_content):
        if idx not in used_german_indices:
             aligned_pairs.append({"english": None, "german": item, "similarity": 0.0})

    # Sort by English page number
    aligned_pairs.sort(key=lambda x: x['english']['page'] if x.get('english') else float('inf'))

    return aligned_pairs

// main.py
import argparse
import time
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import config
from src.processing.json_parser import process_document_json
from src.alignment.semantic_aligner import align_content, ContentItem  # Import ContentItem from here
from src.reporting.markdown_writer import save_to_markdown
from src.reporting.excel_writer import save_alignment_report, save_evaluation_report, save_calculation_report
from src.evaluation.pipeline import run_evaluation_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Aligns and optionally evaluates content from two Azure Document Intelligence JSON files."
    )
    parser.add_argument("english_json", type=str, help="Path to the English JSON file.")
    parser.add_argument("german_json", type=str, help="Path to the German JSON file.")
    parser.add_argument(
        "-o", "--output", type=str, help="Path for the output alignment Excel file.",
        default=None
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run the AI evaluation pipeline after alignment."
    )
    parser.add_argument(
        "--debug-report", action="store_true",
        help="Generate a detailed Excel report showing the score calculations for debugging."
    )
    # New arguments for alignment options
    parser.add_argument(
        "--algorithm", type=str, choices=["mutual", "hungarian"], default="mutual",
        help="Alignment algorithm to use: mutual best match or Hungarian algorithm."
    )
    parser.add_argument(
        "--context-window", type=int, default=0,
        help="Size of context window (0 for no context, 1+ for context-aware embeddings)."
    )
    parser.add_argument(
        "--compare-methods", action="store_true",
        help="Compare different alignment methods and save results."
    )
    args = parser.parse_args()

    # --- 1. Setup Paths ---
    eng_path = Path(args.english_json)
    ger_path = Path(args.german_json)

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_alignment_path = Path(args.output)
    else:
        output_alignment_path = output_dir / f"alignment_{eng_path.stem}_{timestamp}.xlsx"

    output_md_eng_path = output_dir / f"{eng_path.stem}_processed.md"
    output_md_ger_path = output_dir / f"{ger_path.stem}_processed.md"

    # Path for the debug report
    if args.debug_report:
        output_debug_path = output_dir / f"debug_calculations_{eng_path.stem}_{timestamp}.xlsx"
        print(f"Debug Report will be saved to: {output_debug_path}\n")
    else:
        output_debug_path = None

    print("--- Document Alignment Pipeline Started ---")
    print(f"English Source: {eng_path}")
    print(f"German Source:  {ger_path}")
    print(f"Output Alignment Report:  {output_alignment_path}")
    print(f"Alignment Algorithm: {args.algorithm}")
    print(f"Context Window Size: {args.context_window}\n")

    try:
        print("Step 1/5: Processing JSON files...")
        english_content = process_document_json(eng_path)
        german_content = process_document_json(ger_path)
        print(f"-> Extracted {len(english_content)} English segments and {len(german_content)} German segments.\n")
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}")
        return
    except Exception as e:
        print(f"An error occurred during JSON processing: {e}")
        return

    print("Step 2/5: Creating verification Markdown files...")
    save_to_markdown(english_content, output_md_eng_path)
    save_to_markdown(german_content, output_md_ger_path)
    print(f"-> Markdown files saved in '{output_dir.resolve()}'\n")

    # If comparison mode is enabled, run the comparison and exit
    if args.compare_methods:
        print("Running comparison of alignment methods...")
        compare_alignment_methods(
            english_content,
            german_content,
            output_dir,
            eng_path.stem,
            args.evaluate
        )
        return

    print("Step 3/5: Performing semantic alignment...")
    aligned_pairs = align_content(
        english_content,
        german_content,
        algorithm=args.algorithm,
        context_window=args.context_window,
        generate_debug_report=args.debug_report,
        debug_report_path=output_debug_path
    )
    print(f"-> Alignment complete. Found {len(aligned_pairs)} aligned pairs.\n")

    print("Step 4/5: Writing alignment report to Excel...")
    save_alignment_report(aligned_pairs, output_alignment_path)
    print(f"-> Alignment report saved to: {output_alignment_path.resolve()}\n")

    if args.evaluate:
        print("Step 5/5: Running AI evaluation pipeline...")
        try:
            evaluation_results = list(run_evaluation_pipeline(aligned_pairs))

            if not evaluation_results:
                print("-> Evaluation complete. No significant errors were found.")
            else:
                print(f"-> Evaluation complete. Found {len(evaluation_results)} potential errors.")
                output_eval_path = output_dir / f"evaluation_report_{eng_path.stem}_{timestamp}.xlsx"
                save_evaluation_report(evaluation_results, output_eval_path)
                print(f"-> Evaluation report saved to: {output_eval_path.resolve()}")

        except RuntimeError as e:
            print(f"\nERROR: Could not run evaluation. {e}")
            print("Please ensure your AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set in the .env file.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during evaluation: {e}")

    print("\n--- Pipeline Finished Successfully ---")

def compare_alignment_methods(
    english_content: List[ContentItem],
    german_content: List[ContentItem],
    output_dir: Path,
    base_filename: str,
    run_evaluation: bool = False
):
    """
    Compare different alignment methods and save results.

    Args:
        english_content: List of English content items
        german_content: List of German content items
        output_dir: Directory to save output files
        base_filename: Base name for output files
        run_evaluation: Whether to run evaluation on each alignment
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    methods = [
        {"name": "mutual", "context": 0, "desc": "Mutual Best Match"},
        {"name": "hungarian", "context": 0, "desc": "Hungarian Algorithm"},
        {"name": "mutual", "context": 1, "desc": "Mutual Best Match with Context"},
        {"name": "hungarian", "context": 1, "desc": "Hungarian Algorithm with Context"}
    ]

    results = []

    for method in methods:
        print(f"\n--- Testing: {method['desc']} ---")

        # Perform alignment using the current method
        aligned_pairs = align_content(
            english_content,
            german_content,
            algorithm=method["name"],
            context_window=method["context"]
        )

        # Save alignment report
        output_path = output_dir / f"alignment_{base_filename}_{method['name']}_ctx{method['context']}_{timestamp}.xlsx"
        save_alignment_report(aligned_pairs, output_path)
        print(f"-> Alignment report saved to: {output_path}")

        # Calculate statistics
        total_pairs = len(aligned_pairs)
        matched_pairs = sum(1 for pair in aligned_pairs if pair.get('english') and pair.get('german'))
        unmatched_eng = sum(1 for pair in aligned_pairs if pair.get('english') and not pair.get('german'))
        unmatched_ger = sum(1 for pair in aligned_pairs if not pair.get('english') and pair.get('german'))

        # Run evaluation if requested
        eval_errors = 0
        if run_evaluation:
            try:
                print(f"Running evaluation for {method['desc']}...")
                evaluation_results = list(run_evaluation_pipeline(aligned_pairs))
                eval_errors = len(evaluation_results)

                if evaluation_results:
                    eval_path = output_dir / f"eval_{base_filename}_{method['name']}_ctx{method['context']}_{timestamp}.xlsx"
                    save_evaluation_report(evaluation_results, eval_path)
                    print(f"-> Evaluation report saved to: {eval_path}")
            except Exception as e:
                print(f"Evaluation error: {e}")

        # Store results for comparison
        results.append({
            "Method": method['desc'],
            "Total Pairs": total_pairs,
            "Matched Pairs": matched_pairs,
            "Unmatched English": unmatched_eng,
            "Unmatched German": unmatched_ger,
            "Match Rate": f"{matched_pairs/(matched_pairs+unmatched_eng+unmatched_ger):.2%}",
            "Evaluation Errors": eval_errors if run_evaluation else "N/A"
        })

        print(f"-> {method['desc']}: {matched_pairs} matched pairs, {unmatched_eng} unmatched English, {unmatched_ger} unmatched German")

    # Save comparison report
    comparison_df = pd.DataFrame(results)
    comparison_path = output_dir / f"comparison_{base_filename}_{timestamp}.xlsx"
    comparison_df.to_excel(comparison_path, index=False)
    print(f"\nComparison report saved to: {comparison_path}")

    return comparison_df

if __name__ == "__main__":
    main()

