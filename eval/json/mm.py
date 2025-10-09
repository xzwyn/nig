// main.py
# main.py

import argparse
import time
import re
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

import config
from src.processing.json_parser import process_document_json
from src.processing.toc_parser import get_toc_text_from_pdf, structure_toc
from src.alignment.toc_aligner import align_tocs
# The align_content function now returns debug data, so we import it directly
from src.alignment.semantic_aligner import align_content, _get_embeddings_with_context, _calculate_type_matrix, _calculate_proximity_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from src.reporting.excel_writer import save_alignment_report, save_evaluation_report, save_consolidated_debug_report
from src.alignment.semantic_aligner import _get_model # Need access to the model loader

def main():
    parser = argparse.ArgumentParser(
        description="Aligns document content section-by-section using a Table of Contents-first approach."
    )
    # ... (parser arguments are unchanged)
    parser.add_argument("english_pdf", type=str, help="Path to the source English PDF file.")
    parser.add_argument("german_pdf", type=str, help="Path to the source German PDF file.")
    parser.add_argument("english_json", type=str, help="Path to the processed English JSON file.")
    parser.add_argument("german_json", type=str, help="Path to the processed German JSON file.")
    
    parser.add_argument("-o", "--output", type=str, default=None, help="Path for the output alignment Excel file.")
    parser.add_argument("--evaluate", action="store_true", help="Run the AI evaluation pipeline after alignment.")
    parser.add_argument("--debug-report", action="store_true", help="Generate a single detailed score calculation report with a summary and sheets for each section.")
    args = parser.parse_args()

    # --- Setup ---
    # ... (setup is unchanged)
    eng_pdf_path, ger_pdf_path = Path(args.english_pdf), Path(args.german_pdf)
    eng_json_path, ger_json_path = Path(args.english_json), Path(args.german_json)
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("--- Document Alignment Pipeline Started (ToC-First Approach) ---")
    
    # --- Steps 1, 2, 3 are unchanged ---
    print("Step 1/5: Processing full JSON files...")
    full_english_content = process_document_json(eng_json_path)
    full_german_content = process_document_json(ger_json_path)
    
    print("\nStep 2/5: Extracting and structuring Tables of Contents from PDFs...")
    english_toc = structure_toc(get_toc_text_from_pdf(eng_pdf_path, page_num=1))
    german_toc = structure_toc(get_toc_text_from_pdf(ger_pdf_path, page_num=1))

    print("\nStep 3/5: Aligning ToC sections semantically...")
    aligned_sections = align_tocs(english_toc, german_toc)

    # --- Step 4: Perform Section-by-Section Content Alignment ---
    final_aligned_pairs = []
    all_debug_reports = [] # New list to hold data for the consolidated report
    print("\nStep 4/5: Performing content alignment for each matched section...")
    
    model = _get_model(config.MODEL_NAME) # Pre-load model

    # Use enumerate to get a unique index for each section
    for i, section in enumerate(tqdm(aligned_sections, desc="Aligning Sections"), 1):
        eng_sec = section['english']
        ger_sec = section['german']
        
        eng_section_content = [item for item in full_english_content if eng_sec['start_page'] <= item['page'] <= eng_sec['end_page']]
        ger_section_content = [item for item in full_german_content if ger_sec['start_page'] <= item['page'] <= ger_sec['end_page']]

        if not eng_section_content or not ger_section_content:
            continue
        
        # --- Run alignment logic and collect debug data if needed ---
        aligned_pairs, debug_data = align_content(
            english_content=eng_section_content,
            german_content=ger_section_content,
            generate_debug_report=args.debug_report # Pass the flag
        )
        final_aligned_pairs.extend(aligned_pairs)
        
        if args.debug_report and debug_data:
            # Create a unique, clean sheet name
            first_two_words = "_".join(eng_sec['title'].split()[:2])
            sheet_name = f"{i}_{re.sub(r'[^A-Za-z0-9_]', '', first_two_words)} EN"
            
            all_debug_reports.append({
                'sheet_name': sheet_name,
                'data': debug_data
            })

    final_aligned_pairs.sort(key=lambda x: (x['english']['page'] if x.get('english') else float('inf')))
    print(f"-> Alignment complete. Found {len(final_aligned_pairs)} total pairs.\n")

    # --- Step 5: Write Reports ---
    output_alignment_path = Path(args.output) if args.output else output_dir / f"alignment_report_{timestamp}.xlsx"
    print("Step 5/5: Writing final reports...")
    save_alignment_report(final_aligned_pairs, output_alignment_path)
    print(f"-> Alignment report saved to: {output_alignment_path.resolve()}")

    # A single call to the new consolidated debug report writer
    if args.debug_report:
        debug_report_path = output_dir / f"debug_report_{timestamp}.xlsx"
        save_consolidated_debug_report(all_debug_reports, debug_report_path)
        print(f"-> Consolidated debug report saved to: {debug_report_path.resolve()}\n")

    # The evaluation pipeline logic remains unchanged
    if args.evaluate:
        # ... (evaluation code)
        print("Running AI evaluation pipeline...")
        from src.evaluation.pipeline import run_evaluation_pipeline # Import only when needed
        try:
            evaluation_results = list(run_evaluation_pipeline(final_aligned_pairs))
            if evaluation_results:
                print(f"-> Evaluation complete. Found {len(evaluation_results)} potential errors.")
                output_eval_path = output_dir / f"evaluation_report_{timestamp}.xlsx"
                save_evaluation_report(evaluation_results, output_eval_path)
                print(f"-> Evaluation report saved to: {output_eval_path.resolve()}")
            else:
                print("-> Evaluation complete. No significant errors were found.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during evaluation: {e}")


    print("\n--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()

// src/alignment/semantic_aligner.py
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

// src/alignment/toc_aligner.py
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

// src/clients/azure_client.py
import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

_client: Optional[AzureOpenAI] = None
_cfg = {
    "endpoint": None,
    "api_key": None,
    "api_version": None,
    "chat_deployment": None,
    "embedding_deployment": None,
}

def _load_env():
    _cfg["endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    _cfg["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
    _cfg["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    _cfg["chat_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    _cfg["embedding_deployment"] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", _cfg["chat_deployment"])

def _get_client() -> AzureOpenAI:
    global _client
    if _client is not None:
        return _client

    _load_env()
    if not _cfg["endpoint"] or not _cfg["api_key"] or not _cfg["chat_deployment"]:
        raise RuntimeError(
            "Azure OpenAI client is not configured. "
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT in your .env file."
        )

    _client = AzureOpenAI(
        azure_endpoint=_cfg["endpoint"],
        api_key=_cfg["api_key"],
        api_version=_cfg["api_version"],
    )
    return _client

def chat(messages: List[Dict[str, Any]], temperature: float = 0.1, model: Optional[str] = None) -> str:
    client = _get_client()

    deployment = model or _cfg["chat_deployment"]

    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""

def get_embeddings(texts: List[str], model: Optional[str]=None) -> List[List[float]]:
    client = _get_client()
    deployment = model or _cfg['embedding_deployment']
    response = client.embeddings.create(
        input=texts,
        model=deployment
    )
    return [item.embedding for item in response.data]

// src/evaluation/evaluators.py
import json
from src.clients.azure_client import chat  

def evaluate_translation_pair(eng_text: str, ger_text: str, model_name=None):
    prompt = f"""
## ROLE
You are the Primary Translation Auditor for EN→DE corporate reports.
...
(The rest of your detailed prompt for Agent 1 is unchanged)
...
<German Translation>
{ger_text}
</German Translation>

## YOUR RESPONSE
Return the JSON object only—no extra text, no markdown.
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        if j0 != -1 and j1 != -1:
            return json.loads(content[j0:j1])
        return {"error_type": "System Error",
                "explanation": "No JSON object in LLM reply."}
    except Exception as exc:
        print(f"evaluate_translation_pair → {exc}")
        return {"error_type": "System Error", "explanation": str(exc)}

def check_context_mismatch(eng_text: str, ger_text: str, model_name: str = None):
    prompt = f"""
ROLE: Narrative-Integrity Analyst
...
(The rest of your detailed prompt for Agent 3 is unchanged)
...
<German_Translation>
{ger_text}
</German_Translation>
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        return json.loads(content[j0:j1])
    except Exception as exc:
        return {"context_match": "Error", "explanation": str(exc)}

// src/evaluation/pipeline.py
import json
from typing import List, Dict, Any
from tqdm import tqdm

from src.evaluation.evaluators import evaluate_translation_pair, check_context_mismatch
from src.clients.azure_client import chat

AlignedPair = Dict[str, Any]
EvaluationFinding = Dict[str, Any]

def _agent2_validate_finding(
    eng_text: str,
    ger_text: str,
    error_type: str,
    explanation: str,
    model_name: str | None = None,
):
    prompt = f"""
## ROLE
**Senior Quality Reviewer** – you are the final gatekeeper of EN→DE
...
(The rest of your detailed prompt for Agent 2 is unchanged)
...
## YOUR RESPONSE
Return the JSON object only – no extra text.
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        )
        j0, j1 = content.find("{"), content.rfind("}") + 1
        verdict_json = json.loads(content[j0:j1])
        is_confirmed = verdict_json.get("verdict", "").lower() == "confirm"
        reasoning = verdict_json.get("reasoning", "")
        return is_confirmed, reasoning, content.strip()
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"  - Agent-2 JSON parse error: {exc}")
        return False, f"System error: {exc}", "{}"
    except Exception as exc:
        print(f"  - Agent-2 unexpected error: {exc}")
        return False, "System error (non-parsing issue)", "{}"

def run_evaluation_pipeline(aligned_pairs: List[AlignedPair]) -> List[EvaluationFinding]:
    findings = []
    
    for pair in tqdm(aligned_pairs, desc="Evaluating Pairs"):
        eng_elem = pair.get('english')
        ger_elem = pair.get('german')

        if eng_elem and not ger_elem:
            findings.append({
                "type": f"Omission",
                "english_text": eng_elem['text'],
                "german_text": "---",
                "suggestion": "This content from the English document is missing in the German document.",
                "page": eng_elem['page']
            })
            continue
        
        if not eng_elem and ger_elem:
            findings.append({
                "type": f"Addition",
                "english_text": "---",
                "german_text": ger_elem['text'],
                "suggestion": "This content from the German document does not appear to have a source in the English document.",
                "page": ger_elem['page']
            })
            continue

        if eng_elem and ger_elem:
            eng_text = eng_elem['text']
            ger_text = ger_elem['text']

            # Agent 1
            finding = evaluate_translation_pair(eng_text, ger_text)
            error_type = finding.get("error_type", "None")

            if error_type not in ["None", "System Error"]:
                # Agent 2
                is_confirmed, reasoning, _ = _agent2_validate_finding(
                    eng_text, ger_text, error_type, finding.get("explanation")
                )

                if is_confirmed:
                    findings.append({
                        "type": error_type,
                        "english_text": eng_text,
                        "german_text": ger_text,
                        "suggestion": finding.get("suggestion"),
                        "page": eng_elem.get('page'),
                        "original_phrase": finding.get("original_phrase"),
                        "translated_phrase": finding.get("translated_phrase")
                    })
                else: # Agent 2 rejected, run Agent 3
                    context_result = check_context_mismatch(eng_text, ger_text)
                    context_match_verdict = context_result.get('context_match', 'Error')
                    if context_match_verdict.lower() == "no":
                        findings.append({
                            "type": "Context Mismatch",
                            "english_text": eng_text,
                            "german_text": ger_text,
                            "suggestion": context_result.get("explanation"),
                            "page": eng_elem.get('page')
                        })

    return findings

// src/processing/json_parser.py
import json
from pathlib import Path
from typing import List, Dict, Any
import re # <--- NEW IMPORT

import config

# A type alias for our structured content for clarity
ContentItem = Dict[str, Any]

def _convert_table_to_markdown(table_obj: Dict) -> str:
    """Converts an Azure table object into a Markdown string."""
    # ... (function body remains unchanged) ...
    markdown_str = ""
    if not table_obj.get('cells'):
        return ""

    # Create header
    header_cells = [cell for cell in table_obj['cells'] if cell.get('kind') == 'columnHeader']
    if header_cells:
        header_cells.sort(key=lambda x: x['columnIndex'])
        # Handle cells that might span multiple columns
        header_content = []
        for cell in header_cells:
            content = cell.get('content', '').strip()
            col_span = cell.get('columnSpan', 1)
            header_content.extend([content] * col_span)
        
        header_row = "| " + " | ".join(header_content) + " |"
        separator_row = "| " + " | ".join(["---"] * len(header_content)) + " |"
        markdown_str += header_row + "\n" + separator_row + "\n"

    # Create body rows
    body_cells = [cell for cell in table_obj['cells'] if cell.get('kind') is None]
    
    rows = {}
    for cell in body_cells:
        row_idx = cell.get('rowIndex', 0)
        if row_idx not in rows:
            rows[row_idx] = []
        rows[row_idx].append(cell)

    for row_idx in sorted(rows.keys()):
        row_cells = sorted(rows[row_idx], key=lambda x: x.get('columnIndex', 0))
        row_str = "| " + " | ".join([cell.get('content', '').strip() for cell in row_cells]) + " |"
        markdown_str += row_str + "\n"
        
    return markdown_str.strip()


def extract_canonical_toc_headings(full_text: str) -> List[str]:
    """
    Extracts the definite, canonical section headings found in the TOC.
    This directly implements the user's PyMuPDF/regex logic to get the list of keys.
    """
    # Regex patterns provided by the user, modified slightly for robustness:
    # 1. Main sections (A, B, C, D) e.g., 'A _ To Our Investors' (Must capture only up to the first space after the prefix)
    main_title_pattern = re.compile(r"^[A-D] _? (.*)")
    # 2. Numbered sections (e.g., '2 Supervisory Board Report') - Captures the heading text only
    sub_section_pattern = re.compile(r"^\d+ (.*)")
    # 3. All-caps headings (e.g., 'FINANCIAL STATEMENTS') - Captures all caps words followed by non-newline characters
    heading_pattern = re.compile(r"^[A-Z\s]+$")

    canonical_headings = []
    
    # Analyze only the first 5000 characters to capture the TOC, avoiding large text analysis
    toc_text_block = full_text[:5000]

    for line in toc_text_block.split('\n'):
        line = line.strip()
        if not line or len(line) < 5: # Ignore very short lines/noise
            continue

        match = None
        # Check for A/B/C/D sections
        if re.match(r"^[A-D] _", line):
            # Clean the line by stripping everything after the section name (e.g., page numbers, page ranges)
            # Find the index of the first digit (page number) to cut the string
            first_digit_index = next((i for i, char in enumerate(line) if char.isdigit()), len(line))
            
            # Keep only the structural prefix and the section name
            cleaned_line = line[:first_digit_index].strip()
            
            # Strip trailing garbage/page markers that might not be digits
            cleaned_line = cleaned_line.replace('Pages', '').replace('Seiten', '').strip()
            
            # Only accept lines that look like a clean major A/B/C/D heading
            if re.match(r"^[A-D] _ [A-Za-z\s]+$", cleaned_line):
                 match = cleaned_line
                 
        # Check for Numbered Sub-Sections (e.g., 2, 8, 11)
        elif re.match(r"^\d+\s", line):
            # Use the regex to clean out numbers and trailing page numbers/ranges (e.g., "49 Nature of Operations...Pages 49")
            # We assume the heading ends before the page number or the word "Pages/Seiten"
            # In the user's list: "2 Supervisory Board Report", "8 Mandates..."
            parts = line.split()
            if len(parts) > 1 and parts[0].isdigit():
                heading_text = " ".join(parts[1:])
                # Stop at the first occurrence of "Pages" or a page number indicator
                final_heading = heading_text.split('Pages')[0].split('Seiten')[0].strip()
                # Also strip any trailing digits that might be page numbers without the word 'Pages'
                final_heading = re.sub(r'\s*\d+([\s\-\d]+)?$', '', final_heading).strip()
                match = final_heading
                
        # Check for All-Caps Headings (e.g., FINANCIAL STATEMENTS)
        elif heading_pattern.match(line) and len(line) > 5 and ' ' in line:
            match = line
            
        if match and match not in canonical_headings:
            canonical_headings.append(match)
            
    return canonical_headings


def process_document_json(filepath: Path) -> List[ContentItem]:
    """
    Reads and processes an Azure Document Intelligence JSON file,
    now with dedicated handling for tables.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"The file '{filepath}' was not found.")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        analyze_result = data['analyzeResult']
        full_text_content = analyze_result['content']
        raw_paragraphs = analyze_result.get('paragraphs', [])
        pages = analyze_result.get('pages', [])
        raw_tables = analyze_result.get('tables', [])
    except KeyError as e:
        raise ValueError(f"JSON file '{filepath}' is missing expected key: {e}") from e
    
    # --- Step 1: Identify all character offsets belonging to tables to avoid duplication ---
    # ... (rest of Step 1 remains unchanged)
    table_offsets = set()
    for table in raw_tables:
        for span in table.get('spans', []):
            for i in range(span['offset'], span['offset'] + span['length']):
                table_offsets.add(i)
    
    # Identify all character offsets that are handwritten
    handwritten_offsets = set()
    if 'styles' in analyze_result:
        for style in analyze_result['styles']:
            if style.get('isHandwritten') and style.get('spans'):
                for span in style['spans']:
                    for i in range(span['offset'], span['offset'] + span['length']):
                        handwritten_offsets.add(i)
    
    # Create a quick lookup for page number by span offset
    page_lookup = {}
    for page in pages:
        for span in page.get('spans', []):
            for i in range(span['offset'], span['offset'] + span['length']):
                page_lookup[i] = page.get('pageNumber', 0)

    # --- Step 2: Extract all content, including tables, and sort by position ---
    all_content: List[ContentItem] = []

    # Process PARAGRAPHS
    for p in raw_paragraphs:
        role = p.get('role', 'paragraph')
        if role in config.IGNORED_ROLES or not p.get('spans'):
            continue
        
        offset = p['spans'][0]['offset']
        # If the paragraph is inside a table or is handwritten, SKIP it.
        if offset in table_offsets or offset in handwritten_offsets:
            continue
            
        length = p['spans'][0]['length']
        text = full_text_content[offset : offset + length].strip()
        page_number = page_lookup.get(offset, 0)
        if text:
            all_content.append({'text': text, 'type': role, 'page': page_number, 'offset': offset})
            
    # Process TABLES
    for table in raw_tables:
        if not table.get('spans'):
            continue
        offset = table['spans'][0]['offset']
        page_number = page_lookup.get(offset, 0)
        markdown_table = _convert_table_to_markdown(table)
        if markdown_table:
            all_content.append({'text': markdown_table, 'type': 'table', 'page': page_number, 'offset': offset})

    # Sort all extracted content by its character offset to maintain document order
    all_content.sort(key=lambda x: x['offset'])

    # --- Step 3: Stitch broken paragraphs ---
    final_content: List[ContentItem] = []
    stitched_text = ""
    current_page = 0
    current_type = "paragraph"

    for i, segment in enumerate(all_content):
        # If the current element is a table or a structural heading, finalize the previous stitched text.
        is_standalone = segment['type'] in config.STRUCTURAL_ROLES or segment['type'] == 'table'

        if is_standalone:
            if stitched_text: # Finalize any pending paragraph
                final_content.append({'text': stitched_text, 'type': current_type, 'page': current_page})
                stitched_text = ""
            final_content.append(segment) # Add the standalone item
            continue

        # This logic handles stitching of regular paragraphs
        if not stitched_text: # Start a new paragraph
            stitched_text = segment['text']
            current_page = segment['page']
            current_type = segment['type']
        else:
            # If previous text ends with punctuation, start a new paragraph
            if stitched_text.endswith(('.', '!', '?', ':', '•')):
                final_content.append({'text': stitched_text, 'type': current_type, 'page': current_page})
                stitched_text = segment['text']
                current_page = segment['page']
                current_type = segment['type']
            else: # Continue stitching the current paragraph
                stitched_text += f" {segment['text']}"

    # Add the last stitched paragraph if it exists
    if stitched_text:
        final_content.append({'text': stitched_text, 'type': current_type, 'page': current_page})
        
    return final_content

// src/processing/toc_parser.py
# src/processing/toc_parser.py

import re
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF

# A type alias for a structured ToC item
ToCItem = Dict[str, Any]

def get_toc_text_from_pdf(pdf_path: Path, page_num: int = 1) -> str:
    """Extracts raw text from a specific page of a PDF file."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    try:
        with fitz.open(pdf_path) as doc:
            if page_num < len(doc):
                page = doc.load_page(page_num)
                return page.get_text("text")
            else:
                raise IndexError(f"Page {page_num} does not exist in the document.")
    except Exception as e:
        raise IOError(f"Error opening or reading PDF file '{pdf_path}': {e}") from e

def structure_toc(toc_text: str, page_offset: int = 2) -> List[ToCItem]:
    """
    Parses the raw text of a Table of Contents into a structured list of sections,
    including their titles, start pages, and calculated end pages.

    Args:
        toc_text: The raw text extracted from the ToC page.
        page_offset: The number to add to the page numbers found in the ToC
                     to match the document's actual page numbering (e.g., in the JSON).

    Returns:
        A list of structured ToC items.
    """
    # Regex to find lines that start with a number (page number) followed by text
    section_pattern = re.compile(r"^\s*(\d+)\s+(.*)")
    
    structured_list: List[Dict[str, Any]] = []
    lines = toc_text.split('\n')

    for line in lines:
        line = line.strip()
        match = section_pattern.match(line)
        if match:
            page_number_str, title = match.groups()
            
            # Clean up title by removing any trailing page ranges or extra artifacts
            title = re.sub(r'\s+Pages\s+\d+\s*–\s*\d+', '', title).strip()
            
            if title:  # Ensure the title is not empty after cleaning
                structured_list.append({
                    'title': title,
                    'start_page': int(page_number_str) + page_offset
                })

    if not structured_list:
        return []

    # Calculate the end_page for each section
    for i in range(len(structured_list) - 1):
        # The end page of the current section is one less than the start page of the next
        structured_list[i]['end_page'] = structured_list[i+1]['start_page'] - 1

    # Set the end_page for the very last section to a high number to capture all remaining content
    structured_list[-1]['end_page'] = 999  # Or a more sophisticated document end detection

    return structured_list

// src/reporting/excel_writer.py
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

// src/reporting/markdown_writer.py
from pathlib import Path
from typing import List, Dict, Any

ContentItem = Dict[str, Any]

def save_to_markdown(content: List[ContentItem], filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in content:
            if item['type'] in {'title', 'sectionHeading', 'subheading'}:
                f.write(f"## {item['text']}\n\n")
            else:
                f.write(f"{item['text']}\n\n")

// config.py
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")

IGNORED_ROLES = {'pageHeader', 'pageFooter', 'pageNumber'}
STRUCTURAL_ROLES = {'title', 'sectionHeading'}

W_SEMANTIC = 0.80  # Weight for semantic similarity (cosine score)
W_TYPE = 0.10      # Weight for matching content types (e.g., table vs. table)
W_PROXIMITY = 0.10 # Weight for relative position in the document

TYPE_MATCH_BONUS = 0.1
TYPE_MISMATCH_PENALTY = -0.2

# The minimum blended score for a pair to be considered a match
SIMILARITY_THRESHOLD = 0.7

INPUT_DIR: str = "input"
OUTPUT_DIR: str = "output"

// requirements.txt
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

// test.py
import os
import re
import json
import traceback
from difflib import SequenceMatcher
import pandas as pd

# ---------------------------------------------------------------
# Optional packages (graceful fallbacks)
# ---------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit("❌ PyMuPDF not installed. Run: pip install PyMuPDF")

try:
    from sentence_transformers import SentenceTransformer, util
    HAVE_ST = True
except ImportError:
    print("⚠️ sentence-transformers not installed → using basic heading match.")
    HAVE_ST = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download("punkt", quiet=True)
    HAVE_NLTK = True
except Exception:
    print("⚠️ nltk not found → using regex sentence splitter.")
    HAVE_NLTK = False


# ---------------------------------------------------------------
# STEP 1 – Extract text elements
# ---------------------------------------------------------------
def extract_text_elements(pdf_path: str):
    """Return list of text spans with font info."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    elements = []
    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                y = line["bbox"][1]
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    size = round(span.get("size", 0), 1)
                    font = span.get("font", "")
                    bold = any(tag in font for tag in ["Bold", "Black"])
                    elements.append({
                        "page": page_num,
                        "y": y,
                        "size": size,
                        "font": font,
                        "bold": bold,
                        "text": text
                    })
    elements.sort(key=lambda x: (x["page"], x["y"]))
    return elements


# ---------------------------------------------------------------
# STEP 2 – Infer font hierarchy
# ---------------------------------------------------------------
def infer_font_hierarchy(elements, tolerance: float = 0.8):
    sizes = sorted({e["size"] for e in elements}, reverse=True)
    if not sizes:
        return {}
    levels = {}
    level = 1
    prev = None
    for s in sizes:
        if prev and abs(prev - s) > tolerance:
            level += 1
        levels[s] = level
        prev = s
    return levels


# ---------------------------------------------------------------
# STEP 3 – Group paragraphs under headings
# ---------------------------------------------------------------
def build_hierarchy(elements, font_levels, heading_threshold_level: int = 3):
    sections, stack, content = [], [], []
    max_level = max(font_levels.values()) if font_levels else heading_threshold_level + 1

    for el in elements:
        level = font_levels.get(el["size"], max_level + 1)
        is_heading = el["bold"] or (level <= heading_threshold_level and len(el["text"].split()) <= 10)

        if is_heading:
            # Save previous section
            if stack:
                sections.append({
                    "heading": stack[-1]["heading"],
                    "level": stack[-1]["level"],
                    "parent": stack[-2]["heading"] if len(stack) > 1 else None,
                    "page": stack[-1]["page"],
                    "content": " ".join(content).strip()
                })
                content = []
            while stack and level <= stack[-1]["level"]:
                stack.pop()
            stack.append({"heading": el["text"], "level": level, "page": el["page"]})
        else:
            content.append(el["text"])

    # Flush last section
    if stack:
        sections.append({
            "heading": stack[-1]["heading"],
            "level": stack[-1]["level"],
            "parent": stack[-2]["heading"] if len(stack) > 1 else None,
            "page": stack[-1]["page"],
            "content": " ".join(content).strip()
        })
    return sections


# ---------------------------------------------------------------
# STEP 4 – Alignment logic
# ---------------------------------------------------------------
def confidence_band(score: float) -> str:
    if score >= 0.8:
        return "High"
    elif score >= 0.55:
        return "Medium"
    return "Low"


def align_sections(eng_sections, ger_sections, threshold: float = 0.55):
    """Align English ↔ German headings semantically or via fallback."""
    if not eng_sections or not ger_sections:
        return []

    aligned = []
    if HAVE_ST:
        model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        eng_titles = [s["heading"] for s in eng_sections]
        ger_titles = [s["heading"] for s in ger_sections]
        e_emb = model.encode(eng_titles, convert_to_tensor=True)
        g_emb = model.encode(ger_titles, convert_to_tensor=True)
        sim = util.cos_sim(e_emb, g_emb)
        for i, e in enumerate(eng_sections):
            j = int(sim[i].argmax().item())
            score = float(sim[i][j].item())
            if score >= threshold:
                aligned.append({
                    "heading_en": e["heading"],
                    "heading_de": ger_sections[j]["heading"],
                    "score": round(score, 3),
                    "confidence": confidence_band(score),
                    "level": e["level"],
                    "page_en": e["page"],
                    "page_de": ger_sections[j]["page"],
                    "content_en": e["content"],
                    "content_de": ger_sections[j]["content"]
                })
    else:
        used = set()
        for e in eng_sections:
            best_j, best_score = None, 0.0
            for j, g in enumerate(ger_sections):
                if j in used:
                    continue
                s = SequenceMatcher(None, e["heading"].lower(), g["heading"].lower()).ratio()
                if s > best_score:
                    best_score, best_j = s, j
            if best_j is not None and best_score >= 0.3:
                used.add(best_j)
                aligned.append({
                    "heading_en": e["heading"],
                    "heading_de": ger_sections[best_j]["heading"],
                    "score": round(best_score, 3),
                    "confidence": confidence_band(best_score),
                    "level": e["level"],
                    "page_en": e["page"],
                    "page_de": ger_sections[best_j]["page"],
                    "content_en": e["content"],
                    "content_de": ger_sections[best_j]["content"]
                })
    return aligned


# ---------------------------------------------------------------
# STEP 5 – Chunk text safely
# ---------------------------------------------------------------
def chunk_text(text: str, max_words: int = 300):
    if not text:
        return []
    if HAVE_NLTK:
        sentences = sent_tokenize(text)
    else:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, cur, count = [], [], 0
    for s in sentences:
        w = len(s.split())
        if count + w > max_words and cur:
            chunks.append(" ".join(cur))
            cur, count = [], 0
        cur.append(s)
        count += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_bilingual_json(aligned, max_words=300):
    data = {}
    for sec in aligned:
        en_chunks = chunk_text(sec["content_en"], max_words)
        de_chunks = chunk_text(sec["content_de"], max_words)
        pairs = []
        for i in range(max(len(en_chunks), len(de_chunks))):
            pairs.append({
                "en": en_chunks[i] if i < len(en_chunks) else "",
                "de": de_chunks[i] if i < len(de_chunks) else ""
            })
        data[sec["heading_en"]] = {
            "match_score": sec["score"],
            "confidence": sec["confidence"],
            "level": sec["level"],
            "page_en": sec["page_en"],
            "page_de": sec["page_de"],
            "chunks": pairs
        }
    return data


# ---------------------------------------------------------------
# STEP 6 – Main compare function
# ---------------------------------------------------------------
def compare_pdfs(english_pdf, german_pdf,
                 out_csv="bilingual_comparison.csv",
                 out_json="bilingual_comparison.json"):
    try:
        print("🔍 Extracting English structure...")
        eng_elems = extract_text_elements(english_pdf)
        eng_sections = build_hierarchy(eng_elems, infer_font_hierarchy(eng_elems))

        print("🔍 Extracting German structure...")
        ger_elems = extract_text_elements(german_pdf)
        ger_sections = build_hierarchy(ger_elems, infer_font_hierarchy(ger_elems))

        print("🔗 Aligning sections...")
        aligned = align_sections(eng_sections, ger_sections)

        if not aligned:
            print("⚠️ No aligned sections found.")
            return

        pd.DataFrame(aligned).to_csv(out_csv, index=False)
        json.dump(build_bilingual_json(aligned),
                  open(out_json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        print(f"\n✅ Done!  {len(aligned)} aligned sections written.")
        print(f"📄 CSV:  {os.path.abspath(out_csv)}")
        print(f"📘 JSON: {os.path.abspath(out_json)}")

    except Exception as e:
        print("❌ ERROR:", e)
        print(traceback.format_exc())


# ---------------------------------------------------------------
# STEP 7 – Run example
# ---------------------------------------------------------------
if __name__ == "__main__":
    # ⚙️ Edit these paths for your system
    english_pdf = r"C:\Users\M3ZEDTZ\Downloads\2_en.pdf"
    german_pdf  = r"C:\Users\M3ZEDTZ\Downloads\2_de.pdf"

    compare_pdfs(english_pdf, german_pdf)

// test1.py
import os
import json
import traceback
from difflib import SequenceMatcher
from collections import defaultdict

# Try optional imports and set flags
_have_fitz = True
_have_st = True
_have_nltk = True
try:
    import fitz  # PyMuPDF
except Exception as e:
    _have_fitz = False
    fitz = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    _have_st = False
    SentenceTransformer = None
    util = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Ensure punkt is available (if not, try download; if offline, fallback will handle)
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            _have_nltk = False
except Exception:
    _have_nltk = False
    sent_tokenize = None


def extract_text_elements(pdf_path):
    """
    Returns list of text spans with attributes:
    [{'page':int,'y':float,'size':float,'font':str,'bold':bool,'text':str}, ...]
    """
    if not _have_fitz:
        raise RuntimeError("PyMuPDF (fitz) is required but not installed.")
    doc = fitz.open(pdf_path)
    elements = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                y = line["bbox"][1]
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    size = round(span.get("size", 0.0), 1)
                    font = span.get("font", "")
                    bold = ("Bold" in font or "Black" in font)
                    elements.append({
                        "page": page_num,
                        "y": y,
                        "size": size,
                        "font": font,
                        "bold": bold,
                        "text": text
                    })
    # reading order: page, vertical position
    elements.sort(key=lambda x: (x["page"], x["y"]))
    return elements


def infer_font_hierarchy(elements, tolerance=0.8):
    """
    Map font sizes -> discrete hierarchy levels (1=largest).
    Tolerance merges near sizes into same level.
    """
    sizes = sorted({el["size"] for el in elements if el["size"] > 0}, reverse=True)
    if not sizes:
        return {}
    levels = {}
    current_level = 1
    prev = None
    for s in sizes:
        if prev is None:
            levels[s] = current_level
        elif abs(prev - s) <= tolerance:
            levels[s] = current_level
        else:
            current_level += 1
            levels[s] = current_level
        prev = s
    return levels


def build_hierarchy(elements, font_levels, heading_threshold_level=3, heading_word_limit=12):
    """
    Build flat list of sections with parent references using a stack.
    Returns: [{'heading','level','parent','page','content'}, ...]
    """
    sections = []
    current_stack = []
    content_acc = []

    max_level = max(font_levels.values()) if font_levels else heading_threshold_level + 1

    for el in elements:
        level = font_levels.get(el["size"], max_level + 1)
        is_heading = el["bold"] or (level <= heading_threshold_level and len(el["text"].split()) <= heading_word_limit)

        if is_heading:
            # flush last heading's content
            if current_stack:
                sections.append({
                    "heading": current_stack[-1]["heading"],
                    "level": current_stack[-1]["level"],
                    "parent": current_stack[-2]["heading"] if len(current_stack) > 1 else None,
                    "page": current_stack[-1]["page"],
                    "content": " ".join(content_acc).strip()
                })
                content_acc = []

            # pop until this heading is deeper than stack top
            while current_stack and level <= current_stack[-1]["level"]:
                current_stack.pop()

            current_stack.append({
                "heading": el["text"],
                "level": level,
                "page": el["page"]
            })
        else:
            content_acc.append(el["text"])

    # final flush
    if current_stack:
        sections.append({
            "heading": current_stack[-1]["heading"],
            "level": current_stack[-1]["level"],
            "parent": current_stack[-2]["heading"] if len(current_stack) > 1 else None,
            "page": current_stack[-1]["page"],
            "content": " ".join(content_acc).strip()
        })
    return sections


def chunk_text(text, max_words=300):
    """Chunk by sentence + approximate max_words per chunk. Uses nltk if available."""
    if not text:
        return []
    if _have_nltk and sent_tokenize:
        sents = sent_tokenize(text)
    else:
        # naive fallback: split on sentence-ending punctuation
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    chunks = []
    cur = []
    count = 0
    for s in sents:
        words = len(s.split())
        if count + words > max_words and cur:
            chunks.append(" ".join(cur))
            cur = []
            count = 0
        cur.append(s)
        count += words
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def align_sections(eng_sections, ger_sections, model_name="distiluse-base-multilingual-cased-v2", semantic_threshold=0.55):
    """
    Align headings. Primary method: sentence-transformers (if available).
    Fallback: SequenceMatcher ratio on headings.
    Returns list of aligned records: each has en/de heading, score, pages, contents.
    """
    if _have_st:
        try:
            model = SentenceTransformer(model_name)
            eng_titles = [s["heading"] for s in eng_sections]
            ger_titles = [s["heading"] for s in ger_sections]
            if not eng_titles or not ger_titles:
                return []
            emb_en = model.encode(eng_titles, convert_to_tensor=True)
            emb_de = model.encode(ger_titles, convert_to_tensor=True)
            sim = util.cos_sim(emb_en, emb_de)
            aligned = []
            for i, en in enumerate(eng_sections):
                best_j = int(sim[i].argmax().item())
                score = float(sim[i][best_j].item())
                if score >= semantic_threshold:
                    aligned.append({
                        "heading_en": en["heading"],
                        "heading_de": ger_sections[best_j]["heading"],
                        "score": score,
                        "level": en["level"],
                        "page_en": en["page"],
                        "page_de": ger_sections[best_j]["page"],
                        "content_en": en["content"],
                        "content_de": ger_sections[best_j]["content"]
                    })
            return aligned
        except Exception:
            # If ST fails at runtime, fall back gracefully
            pass

    # fallback: simple string similarity on headings
    aligned = []
    used = set()
    for en in eng_sections:
        best_j = None
        best_score = 0.0
        for j, de in enumerate(ger_sections):
            if j in used:
                continue
            score = SequenceMatcher(None, en["heading"].lower(), de["heading"].lower()).ratio()
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None and best_score >= 0.30:  # permissive threshold
            used.add(best_j)
            aligned.append({
                "heading_en": en["heading"],
                "heading_de": ger_sections[best_j]["heading"],
                "score": best_score,
                "level": en["level"],
                "page_en": en["page"],
                "page_de": ger_sections[best_j]["page"],
                "content_en": en["content"],
                "content_de": ger_sections[best_j]["content"]
            })
    return aligned


def build_bilingual_chunked(aligned_sections, max_words=300):
    """
    Produce dict: heading_en -> {match_score, level, page_en, page_de, chunks: [{en,de}, ...]}
    """
    out = {}
    for sec in aligned_sections:
        en_chunks = chunk_text(sec["content_en"], max_words=max_words)
        de_chunks = chunk_text(sec["content_de"], max_words=max_words)
        pairs = []
        for i in range(max(len(en_chunks), len(de_chunks))):
            pairs.append({
                "en": en_chunks[i] if i < len(en_chunks) else "",
                "de": de_chunks[i] if i < len(de_chunks) else ""
            })
        out[sec["heading_en"]] = {
            "match_score": sec["score"],
            "level": sec["level"],
            "page_en": sec["page_en"],
            "page_de": sec["page_de"],
            "chunks": pairs
        }
    return out


def run_pipeline(english_pdf, german_pdf=None,
                 json_out="/mnt/data/bilingual_chunked.json",
                 csv_out="/mnt/data/bilingual_aligned.csv",
                 eng_sections_out="/mnt/data/eng_sections.csv"):
    """
    Run: extracts sections from English and optional German PDF,
    aligns, chunking, writes CSV/JSON.
    Returns dict summary and paths.
    """
    summary = {"success": False, "messages": [], "outputs": {}}
    try:
        if not os.path.exists(english_pdf):
            summary["messages"].append(f"English PDF not found: {english_pdf}")
            return summary

        # English extraction
        eng_elements = extract_text_elements(english_pdf)
        summary["messages"].append(f"English spans: {len(eng_elements)}")
        eng_levels = infer_font_hierarchy(eng_elements)
        eng_sections = build_hierarchy(eng_elements, eng_levels)

        # Save English sections CSV for immediate review
        try:
            import pandas as pd
            df_eng = pd.DataFrame(eng_sections)
            df_eng["content_length"] = df_eng["content"].apply(len)
            df_eng["content_snippet"] = df_eng["content"].apply(lambda x: x[:250] + "..." if len(x) > 250 else x)
            df_eng.to_csv(eng_sections_out, index=False)
            summary["outputs"]["eng_sections_csv"] = eng_sections_out
        except Exception:
            summary["messages"].append("pandas not available; skipping writing eng sections CSV.")

        if not german_pdf or not os.path.exists(german_pdf):
            summary["messages"].append("German PDF missing; pipeline completed with English-only extraction.")
            summary["success"] = True
            return summary

        # German extraction
        ger_elements = extract_text_elements(german_pdf)
        ger_levels = infer_font_hierarchy(ger_elements)
        ger_sections = build_hierarchy(ger_elements, ger_levels)

        # Align
        aligned = align_sections(eng_sections, ger_sections)
        summary["messages"].append(f"Aligned sections: {len(aligned)}")
        if not aligned:
            summary["messages"].append("No aligned sections (low similarity). Try lowering thresholds or check the PDFs.")

        # Save CSV of aligned pairs
        try:
            import pandas as pd
            df = pd.DataFrame([{
                "heading_en": a["heading_en"],
                "heading_de": a["heading_de"],
                "score": a["score"],
                "level": a["level"],
                "page_en": a["page_en"],
                "page_de": a["page_de"]
            } for a in aligned])
            df.to_csv(csv_out, index=False)
            summary["outputs"]["aligned_csv"] = csv_out
        except Exception:
            summary["messages"].append("pandas not available; skipping writing aligned CSV.")

        # chunk & JSON
        bilingual = build_bilingual_chunked(aligned)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(bilingual, f, ensure_ascii=False, indent=2)
        summary["outputs"]["bilingual_json"] = json_out

        summary["success"] = True
        return summary

    except Exception as e:
        tb = traceback.format_exc()
        summary["error"] = str(e)
        summary["traceback"] = tb
        return summary


if __name__ == "__main__":
    # Example usage:
    # - If German PDF is present, pass its path, else pass None.
    EN_PDF = "/mnt/data/shortened Annual Report 2024 Allianz Group.pdf"
    DE_PDF = "/mnt/data/German_shortened Annual Report 2024 Allianz Group de.pdf"  # set to None if missing

    res = run_pipeline(EN_PDF, DE_PDF)
    print("RESULT:", json.dumps(res, indent=2))

