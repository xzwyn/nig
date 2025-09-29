import json
import os
import re
import pandas as pd

def extract_paragraphs_from_json(json_filename: str) -> list:
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
    
    processed_segments = []

    for p in paragraphs:
        p_role = p.get('role')
        
        if p_role in ignored_roles:
            continue
        
        paragraph_text = ""
        if p['spans']:
            start = p['spans'][0]['offset']
            length = p['spans'][0]['length']
            paragraph_text = content_text[start:start + length].strip()

        if not paragraph_text:
            continue
            
        split_match = re.search(r'([.:?!])\s+([A-Z])', paragraph_text)
        is_short_segment = len(paragraph_text.split()) < 10

        if not p_role in structural_roles and is_short_segment and split_match:
            split_index = split_match.start(1) + 1 
            heading_segment = paragraph_text[:split_index].strip()
            body_segment = paragraph_text[split_index:].strip()

            if heading_segment:
                processed_segments.append({'text': heading_segment, 'role': 'sectionHeading'})
            if body_segment:
                processed_segments.append({'text': body_segment, 'role': 'text'})
        else:
            processed_segments.append({'text': paragraph_text, 'role': p_role or 'text'})


    final_paragraphs = []
    current_body_text = ""

    for segment in processed_segments:
        text = segment['text']
        role = segment['role']

        if role in structural_roles:
            if current_body_text:
                final_paragraphs.append(current_body_text)
                current_body_text = ""
            final_paragraphs.append(text)
            continue
        
        if not current_body_text or re.search(r'[.?!]$', current_body_text.strip()):
            # Start a new paragraph
            if current_body_text:
                final_paragraphs.append(current_body_text)
            current_body_text = text
        else:
            # Previous segment was a fragment (e.g., page break mid-sentence), so stitch it.
            current_body_text += " " + text

    # Add the final paragraph if any pending text remains
    if current_body_text:
        final_paragraphs.append(current_body_text)
        
    return final_paragraphs

def pair_content_to_excel(english_json_filename: str, german_json_filename: str, output_excel_filename: str = "Parallel_Content.xlsx"):
    print(f"Starting extraction for English: '{english_json_filename}'")
    english_paragraphs = extract_paragraphs_from_json(english_json_filename)
    print(f"Extracted {len(english_paragraphs)} English paragraphs.")
    
    print(f"Starting extraction for German: '{german_json_filename}'")
    german_paragraphs = extract_paragraphs_from_json(german_json_filename)
    print(f"Extracted {len(german_paragraphs)} German paragraphs.")

    max_len = max(len(english_paragraphs), len(german_paragraphs))

    english_paragraphs += [''] * (max_len - len(english_paragraphs))
    german_paragraphs += [''] * (max_len - len(german_paragraphs))
    
    df = pd.DataFrame({
        'English': english_paragraphs,
        'German': german_paragraphs
    })

    try:
        df.to_excel(output_excel_filename, index=False)
        print(f"\n created aligned content file: '{output_excel_filename}'")
    except Exception as e:
        print(f"\n Error writing to Excel file '{output_excel_filename}': {e}")


if __name__ == "__main__":    
    pair_content_to_excel(
        english_json_filename="67.pdf.json", 
        german_json_filename="67g.pdf.json",
        output_excel_filename="Document_Alignment.xlsx"
    )
