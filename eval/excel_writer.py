# document_aligner/src/reporting/excel_writer.py

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

import config

# (Type aliases and the first two functions, save_alignment_report and save_evaluation_report, are unchanged)
AlignedPair = Dict[str, Any]
EvaluationFinding = Dict[str, Any]
ContentItem = Dict[str, Any]

def save_alignment_report(aligned_data: List[AlignedPair], filepath: Path) -> None:
    """Saves the document alignment data to an Excel file."""
    if not aligned_data:
        print("Warning: No aligned data to save to Excel.")
        return
    report_data = []
    for pair in aligned_data:
        eng_item = pair.get('english')
        ger_item = pair.get('german')
        report_data.append({
            "English": eng_item.get('text', '') if eng_item else "--- OMITTED ---",
            "German": ger_item.get('text', '') if ger_item else "--- ADDED ---",
            "Similarity": f"{pair.get('similarity', 0.0):.4f}",
            "Type": (eng_item.get('type') if eng_item else ger_item.get('type', 'N/A')),
            "English Page": (eng_item.get('page') if eng_item else 'N/A'),
            "German Page": (ger_item.get('page') if ger_item else 'N/A')
        })
    df = pd.DataFrame(report_data)
    try:
        df.to_excel(filepath, index=False, engine='openpyxl')
    except Exception as e:
        print(f"Error: Could not write alignment report to '{filepath}'. Reason: {e}")

def save_evaluation_report(evaluation_results: List[EvaluationFinding], filepath: Path) -> None:
    """Saves the AI evaluation findings to a separate Excel report."""
    if not evaluation_results:
        print("No evaluation findings to save.")
        return
    evaluation_results.sort(key=lambda x: x.get('page', 0))
    df = pd.DataFrame(evaluation_results)
    desired_columns = [
        "page", "type", "suggestion", "english_text", "german_text", 
        "original_phrase", "translated_phrase"
    ]
    final_columns = [col for col in desired_columns if col in df.columns]
    df = df[final_columns]
    try:
        df.to_excel(filepath, index=False, sheet_name='Evaluation_Findings')
    except Exception as e:
        print(f"Error: Could not write evaluation report to '{filepath}'. Reason: {e}")

def save_calculation_report(
    english_content: List[ContentItem],
    german_content: List[ContentItem],
    blended_matrix: np.ndarray,
    semantic_matrix: np.ndarray,
    type_matrix: np.ndarray,
    proximity_matrix: np.ndarray,
    filepath: Path
):
    """
    Saves a highly detailed, two-sheet Excel report showing all alignment score calculations.
    """
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # --- Process English Sheet ---
            eng_report_data = []
            best_ger_indices = np.argmax(blended_matrix, axis=1)

            for i, item in enumerate(english_content):
                best_match_idx = best_ger_indices[i]
                
                raw_semantic = semantic_matrix[i, best_match_idx]
                raw_type = type_matrix[i, best_match_idx]
                raw_proximity = proximity_matrix[i, best_match_idx]
                
                eng_report_data.append({
                    "Text": item['text'],
                    "Type": item['type'],
                    "Page No": item['page'],
                    
                    # --- Detailed Semantic Columns ---
                    "Raw Semantic": f"{raw_semantic:.4f}",
                    "Semantic Calculation": f"{raw_semantic:.4f} x {config.W_SEMANTIC}",
                    "Weighted Semantic": f"{raw_semantic * config.W_SEMANTIC:.4f}",
                    
                    # --- Detailed Type Columns ---
                    "Raw Type": f"{raw_type:.1f}",
                    "Type Calculation": f"{raw_type:.1f} x {config.W_TYPE}",
                    "Weighted Type": f"{raw_type * config.W_TYPE:.4f}",
                    
                    # --- Detailed Proximity Columns ---
                    "Raw Proximity": f"{raw_proximity:.4f}",
                    "Proximity Calculation": f"{raw_proximity:.4f} x {config.W_PROXIMITY}",
                    "Weighted Proximity": f"{raw_proximity * config.W_PROXIMITY:.4f}",
                    
                    "Total Score": f"{blended_matrix[i, best_match_idx]:.4f}",
                    "Best Match (German)": german_content[best_match_idx]['text']
                })
            
            df_eng = pd.DataFrame(eng_report_data)
            df_eng.to_excel(writer, sheet_name='English Calculations', index=False)

            # --- Process German Sheet ---
            ger_report_data = []
            best_eng_indices = np.argmax(blended_matrix, axis=0)

            for j, item in enumerate(german_content):
                best_match_idx = best_eng_indices[j]

                raw_semantic = semantic_matrix[best_match_idx, j]
                raw_type = type_matrix[best_match_idx, j]
                raw_proximity = proximity_matrix[best_match_idx, j]

                ger_report_data.append({
                    "Text": item['text'],
                    "Type": item['type'],
                    "Page No": item['page'],

                    "Raw Semantic": f"{raw_semantic:.4f}",
                    "Semantic Calculation": f"{raw_semantic:.4f} x {config.W_SEMANTIC}",
                    "Weighted Semantic": f"{raw_semantic * config.W_SEMANTIC:.4f}",

                    "Raw Type": f"{raw_type:.1f}",
                    "Type Calculation": f"{raw_type:.1f} x {config.W_TYPE}",
                    "Weighted Type": f"{raw_type * config.W_TYPE:.4f}",

                    "Raw Proximity": f"{raw_proximity:.4f}",
                    "Proximity Calculation": f"{raw_proximity:.4f} x {config.W_PROXIMITY}",
                    "Weighted Proximity": f"{raw_proximity * config.W_PROXIMITY:.4f}",

                    "Total Score": f"{blended_matrix[best_match_idx, j]:.4f}",
                    "Best Match (English)": english_content[best_match_idx]['text']
                })
                
            df_ger = pd.DataFrame(ger_report_data)
            df_ger.to_excel(writer, sheet_name='German Calculations', index=False)
            
    except Exception as e:
        print(f"Error: Could not write debug calculation report to '{filepath}'. Reason: {e}")
