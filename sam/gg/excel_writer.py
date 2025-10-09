# src/reporting/excel_writer.py

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import re

import config

# Type aliases
AlignedPair = Dict[str, Any]
EvaluationFinding = Dict[str, Any]
ContentItem = Dict[str, Any]

# save_alignment_report and save_evaluation_report remain unchanged
def save_alignment_report(aligned_data: List[AlignedPair], filepath: Path) -> None:
    """Saves the document alignment data to an Excel file."""
    if not aligned_data:
        print("Warning: No aligned data to save to Excel.")
        return
    # ... (rest of the function is unchanged)
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
    # ... (rest of the function is unchanged)
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


def _create_debug_dataframe(debug_data: Dict[str, Any]) -> pd.DataFrame:
    """Helper function to create a debug dataframe from raw calculation data."""
    report_data = []
    
    # Unpack the data for clarity
    english_content = debug_data['english_content']
    german_content = debug_data['german_content']
    blended_matrix = debug_data['blended_matrix']
    semantic_matrix = debug_data['semantic_matrix']
    type_matrix = debug_data['type_matrix']
    proximity_matrix = debug_data['proximity_matrix']

    if not german_content:
        return pd.DataFrame([{"Message": "No German content to compare against in this section."}])

    best_ger_indices = np.argmax(blended_matrix, axis=1)

    for i, item in enumerate(english_content):
        best_match_idx = best_ger_indices[i]
        best_german_match = german_content[best_match_idx]
        
        raw_semantic = semantic_matrix[i, best_match_idx]
        raw_type = type_matrix[i, best_match_idx]
        raw_proximity = proximity_matrix[i, best_match_idx]
        
        report_data.append({
            "English Text": item['text'],
            "English Type": item['type'],
            "English Page": item['page'],
            "Weighted Semantic": f"{raw_semantic * config.W_SEMANTIC:.4f}",
            "Weighted Type": f"{raw_type * config.W_TYPE:.4f}",
            "Weighted Proximity": f"{raw_proximity * config.W_PROXIMITY:.4f}",
            "Total Score": f"{blended_matrix[i, best_match_idx]:.4f}",
            "Best Match (German)": best_german_match['text'],
            "Best Match Type": best_german_match['type'],
            "Best Match Page No": best_german_match['page']
        })
        
    return pd.DataFrame(report_data)

def save_consolidated_debug_report(
    all_debug_data: List[Dict[str, Any]], 
    filepath: Path
):
    """
    Saves a single, consolidated debug report with a summary sheet and individual
    sheets for each section's calculations.
    """
    if not all_debug_data:
        print("No debug data was generated to save.")
        return

    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            all_dfs = []
            
            # First, create all the individual section sheets and collect their dataframes
            for report_info in all_debug_data:
                df = _create_debug_dataframe(report_info['data'])
                df.to_excel(writer, sheet_name=report_info['sheet_name'], index=False)
                all_dfs.append(df)
            
            # Now, create the consolidated summary sheet
            summary_df = pd.concat(all_dfs, ignore_index=True)
            summary_df.sort_values(by="English Page", inplace=True)
            
            # Use 'to_excel' on the writer object to add the summary sheet
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

    except Exception as e:
        print(f"Error: Could not write consolidated debug report to '{filepath}'. Reason: {e}")

# The old save_calculation_report function is no longer needed and can be removed.