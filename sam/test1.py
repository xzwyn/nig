
import os
import json
import traceback
from difflib import SequenceMatcher
from collections import defaultdict

# Try optional imports and set flags
_have_fitz = True
_have_st = True
_have_nltk = True
try:
    import fitz  # PyMuPDF
except Exception as e:
    _have_fitz = False
    fitz = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    _have_st = False
    SentenceTransformer = None
    util = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Ensure punkt is available (if not, try download; if offline, fallback will handle)
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            _have_nltk = False
except Exception:
    _have_nltk = False
    sent_tokenize = None


def extract_text_elements(pdf_path):
    """
    Returns list of text spans with attributes:
    [{'page':int,'y':float,'size':float,'font':str,'bold':bool,'text':str}, ...]
    """
    if not _have_fitz:
        raise RuntimeError("PyMuPDF (fitz) is required but not installed.")
    doc = fitz.open(pdf_path)
    elements = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                y = line["bbox"][1]
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    size = round(span.get("size", 0.0), 1)
                    font = span.get("font", "")
                    bold = ("Bold" in font or "Black" in font)
                    elements.append({
                        "page": page_num,
                        "y": y,
                        "size": size,
                        "font": font,
                        "bold": bold,
                        "text": text
                    })
    # reading order: page, vertical position
    elements.sort(key=lambda x: (x["page"], x["y"]))
    return elements


def infer_font_hierarchy(elements, tolerance=0.8):
    """
    Map font sizes -> discrete hierarchy levels (1=largest).
    Tolerance merges near sizes into same level.
    """
    sizes = sorted({el["size"] for el in elements if el["size"] > 0}, reverse=True)
    if not sizes:
        return {}
    levels = {}
    current_level = 1
    prev = None
    for s in sizes:
        if prev is None:
            levels[s] = current_level
        elif abs(prev - s) <= tolerance:
            levels[s] = current_level
        else:
            current_level += 1
            levels[s] = current_level
        prev = s
    return levels


def build_hierarchy(elements, font_levels, heading_threshold_level=3, heading_word_limit=12):
    """
    Build flat list of sections with parent references using a stack.
    Returns: [{'heading','level','parent','page','content'}, ...]
    """
    sections = []
    current_stack = []
    content_acc = []

    max_level = max(font_levels.values()) if font_levels else heading_threshold_level + 1

    for el in elements:
        level = font_levels.get(el["size"], max_level + 1)
        is_heading = el["bold"] or (level <= heading_threshold_level and len(el["text"].split()) <= heading_word_limit)

        if is_heading:
            # flush last heading's content
            if current_stack:
                sections.append({
                    "heading": current_stack[-1]["heading"],
                    "level": current_stack[-1]["level"],
                    "parent": current_stack[-2]["heading"] if len(current_stack) > 1 else None,
                    "page": current_stack[-1]["page"],
                    "content": " ".join(content_acc).strip()
                })
                content_acc = []

            # pop until this heading is deeper than stack top
            while current_stack and level <= current_stack[-1]["level"]:
                current_stack.pop()

            current_stack.append({
                "heading": el["text"],
                "level": level,
                "page": el["page"]
            })
        else:
            content_acc.append(el["text"])

    # final flush
    if current_stack:
        sections.append({
            "heading": current_stack[-1]["heading"],
            "level": current_stack[-1]["level"],
            "parent": current_stack[-2]["heading"] if len(current_stack) > 1 else None,
            "page": current_stack[-1]["page"],
            "content": " ".join(content_acc).strip()
        })
    return sections


def chunk_text(text, max_words=300):
    """Chunk by sentence + approximate max_words per chunk. Uses nltk if available."""
    if not text:
        return []
    if _have_nltk and sent_tokenize:
        sents = sent_tokenize(text)
    else:
        # naive fallback: split on sentence-ending punctuation
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    chunks = []
    cur = []
    count = 0
    for s in sents:
        words = len(s.split())
        if count + words > max_words and cur:
            chunks.append(" ".join(cur))
            cur = []
            count = 0
        cur.append(s)
        count += words
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def align_sections(eng_sections, ger_sections, model_name="distiluse-base-multilingual-cased-v2", semantic_threshold=0.55):
    """
    Align headings. Primary method: sentence-transformers (if available).
    Fallback: SequenceMatcher ratio on headings.
    Returns list of aligned records: each has en/de heading, score, pages, contents.
    """
    if _have_st:
        try:
            model = SentenceTransformer(model_name)
            eng_titles = [s["heading"] for s in eng_sections]
            ger_titles = [s["heading"] for s in ger_sections]
            if not eng_titles or not ger_titles:
                return []
            emb_en = model.encode(eng_titles, convert_to_tensor=True)
            emb_de = model.encode(ger_titles, convert_to_tensor=True)
            sim = util.cos_sim(emb_en, emb_de)
            aligned = []
            for i, en in enumerate(eng_sections):
                best_j = int(sim[i].argmax().item())
                score = float(sim[i][best_j].item())
                if score >= semantic_threshold:
                    aligned.append({
                        "heading_en": en["heading"],
                        "heading_de": ger_sections[best_j]["heading"],
                        "score": score,
                        "level": en["level"],
                        "page_en": en["page"],
                        "page_de": ger_sections[best_j]["page"],
                        "content_en": en["content"],
                        "content_de": ger_sections[best_j]["content"]
                    })
            return aligned
        except Exception:
            # If ST fails at runtime, fall back gracefully
            pass

    # fallback: simple string similarity on headings
    aligned = []
    used = set()
    for en in eng_sections:
        best_j = None
        best_score = 0.0
        for j, de in enumerate(ger_sections):
            if j in used:
                continue
            score = SequenceMatcher(None, en["heading"].lower(), de["heading"].lower()).ratio()
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None and best_score >= 0.30:  # permissive threshold
            used.add(best_j)
            aligned.append({
                "heading_en": en["heading"],
                "heading_de": ger_sections[best_j]["heading"],
                "score": best_score,
                "level": en["level"],
                "page_en": en["page"],
                "page_de": ger_sections[best_j]["page"],
                "content_en": en["content"],
                "content_de": ger_sections[best_j]["content"]
            })
    return aligned


def build_bilingual_chunked(aligned_sections, max_words=300):
    """
    Produce dict: heading_en -> {match_score, level, page_en, page_de, chunks: [{en,de}, ...]}
    """
    out = {}
    for sec in aligned_sections:
        en_chunks = chunk_text(sec["content_en"], max_words=max_words)
        de_chunks = chunk_text(sec["content_de"], max_words=max_words)
        pairs = []
        for i in range(max(len(en_chunks), len(de_chunks))):
            pairs.append({
                "en": en_chunks[i] if i < len(en_chunks) else "",
                "de": de_chunks[i] if i < len(de_chunks) else ""
            })
        out[sec["heading_en"]] = {
            "match_score": sec["score"],
            "level": sec["level"],
            "page_en": sec["page_en"],
            "page_de": sec["page_de"],
            "chunks": pairs
        }
    return out


def run_pipeline(english_pdf, german_pdf=None,
                 json_out="/mnt/data/bilingual_chunked.json",
                 csv_out="/mnt/data/bilingual_aligned.csv",
                 eng_sections_out="/mnt/data/eng_sections.csv"):
    """
    Run: extracts sections from English and optional German PDF,
    aligns, chunking, writes CSV/JSON.
    Returns dict summary and paths.
    """
    summary = {"success": False, "messages": [], "outputs": {}}
    try:
        if not os.path.exists(english_pdf):
            summary["messages"].append(f"English PDF not found: {english_pdf}")
            return summary

        # English extraction
        eng_elements = extract_text_elements(english_pdf)
        summary["messages"].append(f"English spans: {len(eng_elements)}")
        eng_levels = infer_font_hierarchy(eng_elements)
        eng_sections = build_hierarchy(eng_elements, eng_levels)

        # Save English sections CSV for immediate review
        try:
            import pandas as pd
            df_eng = pd.DataFrame(eng_sections)
            df_eng["content_length"] = df_eng["content"].apply(len)
            df_eng["content_snippet"] = df_eng["content"].apply(lambda x: x[:250] + "..." if len(x) > 250 else x)
            df_eng.to_csv(eng_sections_out, index=False)
            summary["outputs"]["eng_sections_csv"] = eng_sections_out
        except Exception:
            summary["messages"].append("pandas not available; skipping writing eng sections CSV.")

        if not german_pdf or not os.path.exists(german_pdf):
            summary["messages"].append("German PDF missing; pipeline completed with English-only extraction.")
            summary["success"] = True
            return summary

        # German extraction
        ger_elements = extract_text_elements(german_pdf)
        ger_levels = infer_font_hierarchy(ger_elements)
        ger_sections = build_hierarchy(ger_elements, ger_levels)

        # Align
        aligned = align_sections(eng_sections, ger_sections)
        summary["messages"].append(f"Aligned sections: {len(aligned)}")
        if not aligned:
            summary["messages"].append("No aligned sections (low similarity). Try lowering thresholds or check the PDFs.")

        # Save CSV of aligned pairs
        try:
            import pandas as pd
            df = pd.DataFrame([{
                "heading_en": a["heading_en"],
                "heading_de": a["heading_de"],
                "score": a["score"],
                "level": a["level"],
                "page_en": a["page_en"],
                "page_de": a["page_de"]
            } for a in aligned])
            df.to_csv(csv_out, index=False)
            summary["outputs"]["aligned_csv"] = csv_out
        except Exception:
            summary["messages"].append("pandas not available; skipping writing aligned CSV.")

        # chunk & JSON
        bilingual = build_bilingual_chunked(aligned)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(bilingual, f, ensure_ascii=False, indent=2)
        summary["outputs"]["bilingual_json"] = json_out

        summary["success"] = True
        return summary

    except Exception as e:
        tb = traceback.format_exc()
        summary["error"] = str(e)
        summary["traceback"] = tb
        return summary


if __name__ == "__main__":
    # Example usage:
    # - If German PDF is present, pass its path, else pass None.
    EN_PDF = "/mnt/data/shortened Annual Report 2024 Allianz Group.pdf"
    DE_PDF = "/mnt/data/German_shortened Annual Report 2024 Allianz Group de.pdf"  # set to None if missing

    res = run_pipeline(EN_PDF, DE_PDF)
    print("RESULT:", json.dumps(res, indent=2))
