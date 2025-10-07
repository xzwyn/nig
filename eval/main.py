import argparse
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import config
from src.processing.json_parser import process_document_json
from src.alignment.semantic_aligner import align_content
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
    print(f"Output Alignment Report:  {output_alignment_path}\n")

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

    print("Step 3/5: Performing semantic alignment...")
    aligned_pairs = align_content(
        english_content,
        german_content,
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

if __name__ == "__main__":
    main()