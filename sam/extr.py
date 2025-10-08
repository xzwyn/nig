import re
import fitz  # PyMuPDF

def get_section_definitions_from_toc(full_text: str):
    """
    Parses the table of contents text to dynamically find the name, title,
    and the exact page range for each main section (A, B, C, D).
    """
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]

    # Patterns to find the main titles and their page ranges
    main_title_pattern = re.compile(r"^[A-Z]_?\s+(.*)")
    page_range_pattern = re.compile(r"(?:Pages|Seiten)\s*(\d+)\s*–\s*(\d+)")

    # Step 1: Collect all titles and page ranges separately
    main_titles = []
    page_ranges = []

    for line in lines:
        if main_title_pattern.match(line) and not re.match(r"^\d+", line) and not re.fullmatch(r"^(CONTENT|INHALT|[A-Z\s]+)$", line):
            main_titles.append(line)
        
        range_match = page_range_pattern.search(line)
        if range_match:
            page_ranges.append((int(range_match.group(1)), int(range_match.group(2))))

    # Step 2: Assemble the section definitions by pairing them
    sections = []
    if len(main_titles) == len(page_ranges) and len(main_titles) > 0:
        for i, (title_line, (start, end)) in enumerate(zip(main_titles, page_ranges)):
            letter = chr(ord('A') + i)
            sections.append({
                'name': letter,
                'start_page': start,
                'end_page': end
            })
        return sections
    else:
        print(f"Warning: Mismatch found. Titles: {len(main_titles)}, Page Ranges: {len(page_ranges)}. Cannot split dynamically.")
        return []


def split_pdf_by_sections(source_doc: fitz.Document, sections: list, file_prefix: str):
    """
    Splits a source PDF into multiple smaller PDFs based on the dynamically found sections.
    """
    PAGE_OFFSET = 2  # The first 2 pages are cover/ToC
    source_filename = source_doc.name
    print(f"\nProcessing '{source_filename}'...")

    if not sections:
        print("No sections were defined from the ToC, skipping split.")
        return

    for section in sections:
        name = section['name']
        start_logical = section['start_page']
        end_logical = section['end_page']

        # Calculate the zero-based index for the PDF pages
        start_index = start_logical + PAGE_OFFSET - 1
        end_index = end_logical + PAGE_OFFSET - 1
        
        output_filename = f"{file_prefix}_section_{name}.pdf"
        
        # Check if page range is valid
        if start_index >= source_doc.page_count or end_index >= source_doc.page_count:
            print(f"  -> Skipping '{output_filename}'. Page range {start_logical}-{end_logical} is out of bounds.")
            continue
            
        print(f"  -> Creating '{output_filename}' covering logical pages {start_logical}-{end_logical}...")

        new_doc = fitz.open()
        new_doc.insert_pdf(source_doc, from_page=start_index, to_page=end_index)
        new_doc.save(output_filename, garbage=4, deflate=True)
        new_doc.close()

    print(f"Successfully split '{source_filename}' into {len(sections)} sections.")

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- Process English Document ---
    try:
        english_pdf_path = "en-allianz-se-annual-report-2024.pdf"
        with fitz.open(english_pdf_path) as doc:
            toc_text = doc.load_page(1).get_text("text")
            english_sections = get_section_definitions_from_toc(toc_text)
            split_pdf_by_sections(doc, english_sections, "english")
    except Exception as e:
        print(f"An error occurred with the English PDF: {e}")

    print("\n" + "-"*40 + "\n")

    # --- Process German Document ---
    try:
        german_pdf_path = "de-allianz-se-geschaeftsbericht-2024.pdf"
        with fitz.open(german_pdf_path) as doc:
            toc_text = doc.load_page(1).get_text("text")
            german_sections = get_section_definitions_from_toc(toc_text)
            split_pdf_by_sections(doc, german_sections, "german")
    except Exception as e:
        print(f"An error occurred with the German PDF: {e}")
