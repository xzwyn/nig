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