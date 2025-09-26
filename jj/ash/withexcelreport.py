import pdfplumber
import fitz  # PyMuPDF
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ===========================
# STEP 1 - Extract text with pdfplumber
# ===========================
def extract_paragraphs(pdf_path):
    """
    Extracts text from a PDF file and splits into paragraphs.
    Splitting heuristic: double newline or layout breaks.
    """
    paragraphs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                paragraphs.extend(parts)
    return paragraphs


# ===========================
# STEP 2 - Extract style info with PyMuPDF
# ===========================
def extract_styles(pdf_path):
    """
    Uses PyMuPDF to extract text spans with font size and weight info.
    Returns a list of {text, size, bold} per block.
    """
    doc = fitz.open(pdf_path)
    styled_blocks = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        styled_blocks.append({
                            "text": s["text"].strip(),
                            "size": round(s["size"], 1),
                            "bold": "Bold" in s["font"] or "Black" in s["font"]
                        })
    return styled_blocks


# ===========================
# STEP 3 - Improved heading detection
# ===========================
def detect_headings_with_style(styled_blocks, min_size_diff=2.0):
    """
    Detect headings using font size and boldness.
    A heading is text with font size > average + min_size_diff OR bold.
    """
    sizes = [b["size"] for b in styled_blocks if b["text"]]
    avg_size = sum(sizes) / len(sizes)

    headings = []
    for b in styled_blocks:
        if not b["text"]:
            continue
        if (b["size"] >= avg_size + min_size_diff) or b["bold"]:
            headings.append(b["text"])
    return headings


# ===========================
# STEP 4 - Tag structure (combine heuristics + style)
# ===========================
def tag_structure(paragraphs, headings):
    """
    Tags each paragraph as 'heading' if it matches styled heading candidates,
    otherwise 'paragraph'.
    """
    structure = []
    for para in paragraphs:
        if para in headings or (len(para.split()) <= 8 and para.isupper()):
            structure.append({"type": "heading", "text": para})
        else:
            structure.append({"type": "paragraph", "text": para})
    return structure


# ===========================
# STEP 5 - Align headings (semantic similarity)
# ===========================
def align_headings(eng_struct, ger_struct, model):
    eng_headings = [x["text"] for x in eng_struct if x["type"] == "heading"]
    ger_headings = [x["text"] for x in ger_struct if x["type"] == "heading"]

    eng_emb = model.encode(eng_headings, convert_to_tensor=True)
    ger_emb = model.encode(ger_headings, convert_to_tensor=True)
    cosine_scores = util.cos_sim(eng_emb, ger_emb)

    alignments = {}
    for i, eng_h in enumerate(eng_headings):
        best_j = int(cosine_scores[i].argmax())
        score = float(cosine_scores[i][best_j])
        alignments[eng_h] = {"german": ger_headings[best_j], "score": score}

    return alignments


# ===========================
# STEP 6 - Group paragraphs under headings
# ===========================
def group_by_sections(structure):
    grouped = {}
    current_heading = "INTRODUCTION"
    grouped[current_heading] = []

    for item in structure:
        if item["type"] == "heading":
            current_heading = item["text"]
            grouped[current_heading] = []
        else:
            grouped[current_heading].append(item["text"])
    return grouped


# ===========================
# STEP 7 - Compare paragraphs inside sections
# ===========================
def compare_paragraphs(eng_grouped, ger_grouped, heading_matches, model, threshold=0.65):
    issues = []

    for eng_h, match in heading_matches.items():
        ger_h = match["german"]
        eng_paras = eng_grouped.get(eng_h, [])
        ger_paras = ger_grouped.get(ger_h, [])
        if not eng_paras or not ger_paras:
            continue

        eng_emb = model.encode(eng_paras, convert_to_tensor=True)
        ger_emb = model.encode(ger_paras, convert_to_tensor=True)
        sim_matrix = util.cos_sim(eng_emb, ger_emb)

        for i, eng_p in enumerate(eng_paras):
            best_j = int(sim_matrix[i].argmax())
            score = float(sim_matrix[i][best_j])
            if score < threshold:
                issues.append({
                    "section_en": eng_h,
                    "para_en": eng_p,
                    "para_de": ger_paras[best_j],
                    "similarity": score
                })
    return issues


# ===========================
# STEP 8 - Export issues to Excel
# ===========================
def export_to_excel(issues, filename="translation_issues.xlsx"):
    """
    Save issues (low similarity paragraph pairs) into Excel.
    """
    df = pd.DataFrame(issues)
    df.to_excel(filename, index=False)
    print(f"\n✅ Exported {len(df)} issues to {filename}")


# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    english_pdf = "test_1_e.pdf"
    german_pdf = "test_1_g.pdf"

    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    # Extract raw paragraphs
    english_paras = extract_paragraphs(english_pdf)
    german_paras = extract_paragraphs(german_pdf)

    # Extract style info
    eng_styles = extract_styles(english_pdf)
    ger_styles = extract_styles(german_pdf)

    # Detect headings via font size + bold
    eng_headings = detect_headings_with_style(eng_styles)
    ger_headings = detect_headings_with_style(ger_styles)

    # Tag structure
    english_struct = tag_structure(english_paras, eng_headings)
    german_struct = tag_structure(german_paras, ger_headings)

    # Align headings semantically
    heading_matches = align_headings(english_struct, german_struct, model)
    print("\n--- HEADING ALIGNMENTS ---")
    for eng, match in heading_matches.items():
        print(f"[EN] {eng}\n[DE] {match['german']} (score={match['score']:.2f})\n")

    # Group paragraphs
    eng_grouped = group_by_sections(english_struct)
    ger_grouped = group_by_sections(german_struct)

    # Compare paragraphs
    issues = compare_paragraphs(eng_grouped, ger_grouped, heading_matches, model)

    # Export issues to Excel
    export_to_excel(issues, "translation_issues.xlsx")
