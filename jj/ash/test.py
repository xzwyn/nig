import json
import os
import re

def create_markdown_from_doc_json(json_filename="67.pdf.json"):
    """
    Reads content and paragraph structure from an Azure IDP JSON response,
    filters out page furniture, applies scalable hybrid logic to stitch/separate 
    paragraphs, and saves the extracted text to a .md file.
    
    Args:
        json_filename (str): The name of the input JSON file.
    """
    # Define the output Markdown filename based on the input name
    md_filename = os.path.splitext(json_filename)[0] + "_CLEANED_SCALABLE.md"

    # 1. Read the JSON file
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_filename}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_filename}' is not a valid JSON file.")
        return

    # 2. Extract necessary data
    try:
        content_text = data['analyzeResult']['content']
        paragraphs = data['analyzeResult']['paragraphs']
    except KeyError:
        print("Error: Could not find 'analyzeResult', 'content', or 'paragraphs' key in the JSON structure.")
        return

    # Roles to ignore (page furniture)
    ignored_roles = ["pageHeader", "pageFooter", "pageNumber"]
    
    # Roles that should always be treated as separate headings/structural elements
    structural_roles = ['title', 'sectionHeading', 'subheading']
    
    processed_segments = []

    # 3. First Pass: Filter, Clean, and Apply Hybrid Heuristic Split
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
            
        # Heuristic for splitting common merged Heading/Body combinations
        # This handles cases like "OverviewIm Geschäftsjahr..." where IDP missed the role.
        
        # Split point is detected by looking for terminal punctuation (like a period or colon)
        # that appears relatively early in a paragraph segment.
        split_match = re.search(r'([.:?!])\s+([A-Z])', paragraph_text)
        
        # A split is considered necessary if:
        # a) It's a short segment (likely a heading)
        # b) It's not tagged as a structural role already
        # c) A punctuation followed by a capital letter is found early in the segment (e.g., a short heading or list item ended with punctuation)
        is_short_segment = len(paragraph_text.split()) < 10

        if not p_role in structural_roles and is_short_segment and split_match:
            # Split the segment: everything before the punctuation is the heading, the rest is body
            split_index = split_match.start(1) + 1 # Include the punctuation mark
            
            heading_segment = paragraph_text[:split_index].strip()
            body_segment = paragraph_text[split_index:].strip()

            if heading_segment:
                # Treat the split-off heading with a generic heading role if it wasn't recognized
                processed_segments.append({'text': heading_segment, 'role': 'sectionHeading'})
            if body_segment:
                processed_segments.append({'text': body_segment, 'role': 'text'})
        else:
            # Use original text and role, defaulting to 'text' if no role is present
            processed_segments.append({'text': paragraph_text, 'role': p_role or 'text'})


    # 4. Second Pass: Stitch body paragraphs and respect structural elements
    final_paragraphs = []
    current_body_text = ""

    for segment in processed_segments:
        text = segment['text']
        role = segment['role']

        # If the segment is an explicit structural heading, finalize any pending body text and add the heading.
        if role in structural_roles:
            if current_body_text:
                final_paragraphs.append(current_body_text)
                current_body_text = ""
            final_paragraphs.append(text)
            continue
        
        # If it's body text (role == 'text'):
        
        # Check if the current stitched text is empty or ends with terminal punctuation.
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

    # 5. Write the final content to the Markdown file
    try:
        with open(md_filename, 'w', encoding='utf-8') as md_file:
            # Join with two newlines for the requested extra space between paragraphs
            md_file.write("\n\n".join(final_paragraphs).strip())
        
        print(f"Successfully processed and cleaned content from '{json_filename}' using scalable logic.")
        print(f"Saved content to '{md_filename}'.")
        
    except IOError:
        print(f"Error: Could not write to the output file '{md_filename}'.")

if __name__ == "__main__":
    create_markdown_from_doc_json("67.pdf.json")