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
from src.alignment.semantic_aligner import align_content # <-- This is the only function needed from here
from src.reporting.excel_writer import save_alignment_report, save_evaluation_report, save_consolidated_debug_report

def main():
    parser = argparse.ArgumentParser(
        description="Aligns document content section-by-section using a Table of Contents-first approach."
    )
    parser.add_argument("english_pdf", type=str, help="Path to the source English PDF file.")
    parser.add_argument("german_pdf", type=str, help="Path to the source German PDF file.")
    parser.add_argument("english_json", type=str, help="Path to the processed English JSON file.")
    parser.add_argument("german_json", type=str, help="Path to the processed German JSON file.")
    
    parser.add_argument("-o", "--output", type=str, default=None, help="Path for the output alignment Excel file.")
    parser.add_argument("--evaluate", action="store_true", help="Run the AI evaluation pipeline after alignment.")
    parser.add_argument("--debug-report", action="store_true", help="Generate a single detailed score calculation report with a summary and sheets for each section.")
    args = parser.parse_args()

    # --- Setup ---
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
    
    for i, section in enumerate(tqdm(aligned_sections, desc="Aligning Sections"), 1):
        eng_sec = section['english']
        ger_sec = section['german']
        
        eng_section_content = [item for item in full_english_content if eng_sec['start_page'] <= item['page'] <= eng_sec['end_page']]
        ger_section_content = [item for item in full_german_content if ger_sec['start_page'] <= item['page'] <= ger_sec['end_page']]

        if not eng_section_content or not ger_section_content:
            continue
        
        aligned_pairs, debug_data = align_content(
            english_content=eng_section_content,
            german_content=ger_section_content,
            generate_debug_report=args.debug_report
        )
        final_aligned_pairs.extend(aligned_pairs)
        
        if args.debug_report and debug_data:
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

    if args.debug_report:
        debug_report_path = output_dir / f"debug_report_{timestamp}.xlsx"
        save_consolidated_debug_report(all_debug_reports, debug_report_path)
        print(f"-> Consolidated debug report saved to: {debug_report_path.resolve()}\n")

    if args.evaluate:
        print("Running AI evaluation pipeline...")
        from src.evaluation.pipeline import run_evaluation_pipeline
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
