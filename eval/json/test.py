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
