# main.py
import json
import os
import re
from dotenv import load_dotenv
from agent import run_evaluation_pipeline

# Load environment variables from .env file for the Azure OpenAI client
load_dotenv(override=True)

def extract_elements_from_json(json_filename: str) -> list:
    """
    Processes a JSON file from Azure AI Document Intelligence, extracting and structuring
    paragraphs, headings, and other content elements.

    This function is adapted from your provided logic to produce a list of dictionaries
    compatible with the agentic evaluation pipeline.
    """
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_filename}' was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file '{json_filename}' is not a valid JSON file.")
        return []

    try:
        content_text = data['analyzeResult']['content']
        paragraphs = data['analyzeResult']['paragraphs']
    except KeyError:
        print("Error: Could not find 'analyzeResult', 'content', or 'paragraphs' key in the JSON structure.")
        return []

    ignored_roles = ["pageHeader", "pageFooter", "pageNumber"]
    structural_roles = ['title', 'sectionHeading', 'subheading']
    
    # This list will hold dictionaries for each processed content block.
    processed_elements = []
    element_id_counter = 0

    for p in paragraphs:
        p_role = p.get('role')
        
        if p_role in ignored_roles:
            continue
        
        paragraph_text = ""
        # NOTE: The Azure JSON does not easily provide a page number per paragraph.
        # We will use the element index as a proxy for location.
        page_num = p.get('boundingRegions', [{}])[0].get('pageNumber', 0)

        if p.get('spans'):
            start = p['spans'][0]['offset']
            length = p['spans'][0]['length']
            paragraph_text = content_text[start:start + length].strip()

        if not paragraph_text:
            continue
        
        # Determine the element type for the evaluation pipeline
        element_type = "heading" if p_role in structural_roles else "paragraph"
        
        processed_elements.append({
            'id': f"e_{element_id_counter}",
            'text': paragraph_text,
            'role': p_role or 'text',
            'type': element_type,
            'page': page_num
        })
        element_id_counter += 1

    return processed_elements


def align_documents_from_json(english_json_filename: str, german_json_filename: str) -> list:
    """
    Extracts elements from both English and German JSON files and performs a
    sequential alignment, creating pairs for the evaluation pipeline.
    """
    print(f"\n--- Document Extraction Stage (from Azure JSON) ---")
    print(f"Extracting English content from: '{english_json_filename}'")
    eng_elements = extract_elements_from_json(english_json_filename)
    print(f"Extracted {len(eng_elements)} English elements.")
    
    print(f"Extracting German content from: '{german_json_filename}'")
    ger_elements = extract_elements_from_json(german_json_filename)
    print(f"Extracted {len(ger_elements)} German elements.")

    # Perform sequential pairing
    aligned_pairs = []
    max_len = max(len(eng_elements), len(ger_elements))

    for i in range(max_len):
        eng_elem = eng_elements[i] if i < len(eng_elements) else None
        ger_elem = ger_elements[i] if i < len(ger_elements) else None
        aligned_pairs.append((eng_elem, ger_elem))
    
    print(f"\n--- Alignment complete ---")
    print(f"Total pairs produced: {len(aligned_pairs)}")
    return aligned_pairs


if __name__ == "__main__":
    # Define the input files from Azure Document Intelligence
    eng_json = "67.pdf.json"
    ger_json = "67g.pdf.json"

    # 1. Align documents using the new JSON-based logic
    aligned_pairs = align_documents_from_json(eng_json, ger_json)

    # 2. Run the unchanged agentic evaluation pipeline
    final_results = []
    # The evaluation function is a generator, so we iterate through it
    for result in run_evaluation_pipeline(aligned_pairs):
        final_results.append(result)
        print("\n--- Finding Reported ---")
        print(json.dumps(result, indent=2))
        print("------------------------")

    print("\n--- FINAL SUMMARY ---")
    if not final_results:
        print("✅ No significant errors were found.")
    else:
        print(f"Found {len(final_results)} total issues.")
        # print(json.dumps(final_results, indent=2))
