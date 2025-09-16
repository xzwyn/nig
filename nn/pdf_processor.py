# pdf_processor.py
# ---------------------------------------------------------------------
#  OFF-LINE PDF TEXT-EXTRACTION WITH OPTIONAL OCR FALL-BACK
# ---------------------------------------------------------------------
import io
import re
import traceback
from typing import List, Dict, Any, Tuple, Set

import fitz                              # PyMuPDF
import pytesseract                       # local OCR (Tesseract)
from PIL import Image                    # convert pixmap → PIL.Image

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title,
    NarrativeText,
    ListItem,
    Text,
    Table,
)

# ---------------------------------------------------------------------
#  1.  HELPER UTILITIES
# ---------------------------------------------------------------------
def map_element_type(element) -> str:
    if isinstance(element, Title):
        return "heading"
    if isinstance(element, Table):
        return "table"
    if isinstance(element, NarrativeText):
        return "paragraph"
    if isinstance(element, ListItem):
        return "list_item"
    if isinstance(element, Text):
        return "paragraph"
    return "paragraph"


def _log_err(prefix: str, exc: Exception):
    print(f"{prefix}: {exc}")
    print(traceback.format_exc())


def _get_page_count(pdf_path: str) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


# ---------------------------------------------------------------------
#  2.  OCR FALL-BACK FOR A SINGLE PAGE
# ---------------------------------------------------------------------
def _ocr_page_to_text(page: fitz.Page, lang: str = "eng") -> str:
    """
    Render page → PNG (300 dpi) → Tesseract → text.
    """
    pix = page.get_pixmap(dpi=300)           # high-res raster
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()


# ---------------------------------------------------------------------
#  3.  PYMuPDF BLOCK PARSING (WITH OPTIONAL OCR)
# ---------------------------------------------------------------------
def _page_blocks_with_pymupdf(
    pdf_path: str,
    one_based_page: int,
    lang: str,
    enable_ocr_if_empty: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract text blocks from a page.  If no selectable text is present
    and `enable_ocr_if_empty` is True, run Tesseract to obtain the text.
    """
    items: List[Dict[str, Any]] = []

    with fitz.open(pdf_path) as doc:
        p0 = one_based_page - 1
        if p0 < 0 or p0 >= doc.page_count:
            return items
        page = doc[p0]

        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])

        # OCR fallback when the PDF page contains no real text layer
        if not blocks:
            if enable_ocr_if_empty:
                ocr_txt = _ocr_page_to_text(page, lang)
                if ocr_txt:
                    blocks = [{"lines": [{"spans": [{"text": ocr_txt, "size": 10.0}]}]}]

        # ---------- derive heading threshold (font size heuristic) ----------
        sizes = [
            span.get("size", 0.0)
            for block in blocks
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        if sizes:
            sizes_sorted = sorted(sizes)
            p80 = sizes_sorted[int(0.8 * (len(sizes_sorted) - 1))]
            heading_thresh = max(p80, (sum(sizes) / len(sizes)) + 2.0)
        else:
            heading_thresh = 1e9

        # ---------- build elements ----------
        for block in blocks:
            block_txt = ""
            max_sz = 0.0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    if txt:
                        block_txt += txt
                        max_sz = max(max_sz, float(span.get("size", 0.0)))
                block_txt += "\n"

            block_txt = block_txt.strip()
            if not block_txt:
                continue

            first_line = block_txt.splitlines()[0].strip()
            is_list = first_line.startswith(("-", "•")) or first_line[:3].strip().rstrip(".").isdigit()

            typ = (
                "heading"
                if max_sz >= heading_thresh and len(block_txt.split()) <= 20
                else "list_item"
                if is_list
                else "paragraph"
            )

            items.append(
                {
                    "id": "tmp",
                    "text": block_txt,
                    "type": typ,
                    "page": one_based_page,
                }
            )
    return items

def extract_structural_elements(
    pdf_path: str,
    language: str,
    strategy: str = "fast",
    prefer_tables: bool = True,
    ensure_page_coverage: bool = True,
) -> List[Dict[str, Any]]:
    """
    1. Try Un-structured’s built-in engines (`strategy`, optionally OCR).
    2. Guarantee every page is represented; use PyMuPDF + local OCR for
       pages that contain no selectable text at all.
    """
    print(f"\n--- PDF Processing Stage (strategy='{strategy}', language='{language}') ---")

    # ---------- Step 1 : Try unstructured.partition.pdf ----------
    try_order: List[Tuple[str, bool]] = []
    try_order.append((strategy, bool(prefer_tables)))

    # Allow a pure OCR pass *after* fast if desired
    if strategy != "ocr_only":
        try_order.append(("ocr_only", False))          # <-- *new* fall-back pass

    raw_elements = None
    final_elements: List[Dict[str, Any]] = []

    for strat, infer_tbl in try_order:
        try:
            print(f"Trying unstructured.partition.pdf with strategy='{strat}', infer_table_structure={infer_tbl} ...")
            raw_elements = partition_pdf(
                filename=pdf_path,
                strategy=strat,
                infer_table_structure=infer_tbl,
                languages=[language],
            )
            final_elements = _elements_to_final(raw_elements)
            print(f"Processing complete. Found {len(final_elements)} classified elements.")
            break
        except Exception as exc:
            _log_err(f"Unstructured failed on strategy='{strat}'", exc)

    # ---------- Step 2 : Fallback if Un-structured completely failed ----------
    if raw_elements is None:
        print("All unstructured strategies failed – using PyMuPDF + OCR for the entire document.")
        total_pages = _get_page_count(pdf_path)
        final_elements = []
        for p in range(1, total_pages + 1):
            final_elements.extend(
                _page_blocks_with_pymupdf(pdf_path, p, lang=language[:3])
            )
        _reindex(final_elements)
        print(f"Full document fill complete. Elements: {len(final_elements)}")
        return final_elements

    # ---------- Step 3 : Ensure every physical page is represented ----------
    if ensure_page_coverage:
        covered_pages = _raw_pages_present(raw_elements)
        total_pages = _get_page_count(pdf_path)
        missing_pages = sorted(set(range(1, total_pages + 1)) - covered_pages)

        if missing_pages:
            print(f"Page coverage gap detected → filling via PyMuPDF + OCR (pages {missing_pages}) ...")
            for p in missing_pages:
                final_elements.extend(
                    _page_blocks_with_pymupdf(pdf_path, p, lang=language[:3])
                )
            _reindex(final_elements)
            print(f"After gap-fill total elements: {len(final_elements)}")

    return final_elements


# ---------------------------------------------------------------------
#  5.  SUPPORT FUNCTIONS (UNCHANGED BELOW EXCEPT _reindex)
# ---------------------------------------------------------------------
def _raw_pages_present(raw_elements) -> Set[int]:
    pages = set()
    for el in raw_elements:
        try:
            pg = getattr(getattr(el, "metadata", None), "page_number", None)
            if isinstance(pg, int) and pg > 0:
                pages.add(pg)
        except Exception:
            pass
    return pages


def _elements_to_final(raw_elements) -> List[Dict[str, Any]]:
    final: List[Dict[str, Any]] = []
    idx = 0
    for element in raw_elements:
        try:
            text = (getattr(element, "text", "") or "").strip()
            if not text:
                continue
            typ = map_element_type(element)
            page_num = getattr(getattr(element, "metadata", None), "page_number", None)
            page = page_num if isinstance(page_num, int) and page_num > 0 else 0
            item = {
                "id": f"e_{idx}",
                "text": text,
                "type": typ,
                "page": page,
            }
            if typ == "table":
                item["html"] = getattr(getattr(element, "metadata", None), "text_as_html", "") or ""
            final.append(item)
            idx += 1
        except Exception as exc:
            _log_err("Skipping an element", exc)
    return final


def _reindex(elements: List[Dict[str, Any]]):
    elements.sort(key=lambda x: (x.get("page", 0)))
    for i, el in enumerate(elements):
        el["id"] = f"e_{i}"
