import os
import re
from langchain_community.document_loaders import PyPDFLoader

# --- Configuration ---
FILE_NAME = "67g.pdf" # Make sure this file is in the same directory

# --- Configuration for Header/Footer Removal ---
# The script will remove any line that STARTS WITH the text below.
# Set to "" to disable.
HEADER_STARTS_WITH = "A _ An unsere Aktionärinnen und Aktionäre"

# The script will remove any line that CONTAINS the text below.
# Set to "" to disable.
FOOTER_CONTAINS = "Geschäftsbericht 2024 − Allianz Konzern"


def remove_headers_and_footers(text, header_start, footer_content):
    """
    Removes header and footer lines from a page's text.
    """
    # Split the page text into individual lines
    lines = text.split('\n')
    
    # Keep only the lines that are NOT headers or footers
    kept_lines = []
    for line in lines:
        stripped_line = line.strip()
        
        # Check if the line matches the header or footer pattern
        is_header = stripped_line.startswith(header_start) if header_start else False
        is_footer = footer_content in stripped_line if footer_content else False
        
        # If it's neither a header nor a footer, keep it
        if not is_header and not is_footer:
            kept_lines.append(line)
            
    # Join the kept lines back into a single block of text
    return "\n".join(kept_lines)


def stitch_paragraphs(full_text):
    """
    Cleans and stitches text to correctly form paragraphs.
    """
    # Re-join hyphenated words (e.g., "Konzernlage-\nbericht" -> "Konzernlagebericht")
    processed_text = re.sub(r'-\n', '', full_text)
    
    # Split the text into paragraphs based on one or more empty lines
    paragraphs = re.split(r'\n\s*\n', processed_text)
    
    cleaned_paragraphs = []
    for para in paragraphs:
        # For each paragraph, replace single newlines with a space
        cleaned_para = para.replace('\n', ' ').strip()
        
        if cleaned_para:
            cleaned_paragraphs.append(cleaned_para)
            
    return cleaned_paragraphs


# --- Main Execution ---
if not os.path.exists(FILE_NAME):
    print(f"Error: The file '{FILE_NAME}' was not found.")
else:
    try:
        loader = PyPDFLoader(FILE_NAME)
        pages = loader.load()

        # >>> NEW STEP: Clean headers and footers from each page before combining <<<
        cleaned_page_texts = []
        for page in pages:
            cleaned_content = remove_headers_and_footers(
                page.page_content,
                HEADER_STARTS_WITH,
                FOOTER_CONTAINS
            )
            cleaned_page_texts.append(cleaned_content)

        # Combine the CLEANED text from all pages into one single string
        full_document_text = "".join(cleaned_page_texts)

        # Apply the logic to stitch and clean paragraphs
        final_paragraphs = stitch_paragraphs(full_document_text)

        print(f"--- Successfully extracted and cleaned text from {FILE_NAME} ---\n")

        # Print the final, correctly joined paragraphs
        for i, para in enumerate(final_paragraphs):
            print(f"[Paragraph {i + 1}]")
            print(para)
            print("-" * 20)

    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")