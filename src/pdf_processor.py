# pdf_processor.py
# ---------------------------------------------------------------------
#  DOC EXTRACTION VIA DOCLING + GRANITE-DOCLING-258M WITH TYPE OVERLAY
#  USING UNSTRUCTURED.partition_pdf FOR BETTER ITEM CLASSIFICATION
# ---------------------------------------------------------------------
import traceback
from typing import List, Dict, Any, Tuple, Set
from difflib import SequenceMatcher
import re

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title,
    NarrativeText,
    ListItem,
    Text,
    Table,
)

from docling_client import run_docling_granite


def _log_err(prefix: str, exc: Exception):
    print(f"{prefix}: {exc}")
    print(traceback.format_exc())


def _normalize(txt: str) -> str:
    # Normalize text for fuzzy matching
    t = txt or ""
    t = t.lower()
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _collect_partition_texts(raw_elements) -> Dict[int, Dict[str, Set[str]]]:
    """
    Build per-page sets of normalized texts by element type from Unstructured.
    This helps classify Docling segments more accurately.
    """
    per_page: Dict[int, Dict[str, Set[str]]] = {}
    for el in raw_elements:
        try:
            pg = getattr(getattr(el, "metadata", None), "page_number", None)
            if not isinstance(pg, int) or pg <= 0:
                continue
            txt = (getattr(el, "text", "") or "").strip()
            if not txt:
                continue
            n = _normalize(txt)
            bucket = per_page.setdefault(pg, {"title": set(), "list": set(), "para": set(), "table": set()})
            if isinstance(el, Title):
                bucket["title"].add(n)
            elif isinstance(el, ListItem):
                bucket["list"].add(n)
            elif isinstance(el, Table):
                bucket["table"].add(n)  # rarely used; docling already outputs tables
            elif isinstance(el, (NarrativeText, Text)):
                bucket["para"].add(n)
        except Exception:
            continue
    return per_page


def _is_similar(a: str, b: str, thr: float = 0.85) -> bool:
    if not a or not b:
        return False
    # Allow contains match for short headings/list items
    if len(a) < 60 and (a in b or b in a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= thr


def _overlay_types_with_partition(docling_elements: List[Dict[str, Any]], raw_elements) -> List[Dict[str, Any]]:
    """
    For each Docling element, refine its type using partition_pdf-derived
    texts on the same page (titles, list items). Only adjusts when there
    is a confident match; otherwise leaves Docling type as-is.
    """
    per_page_sets = _collect_partition_texts(raw_elements)

    refined: List[Dict[str, Any]] = []
    for el in docling_elements:
        typ = el.get("type", "paragraph")
        page = el.get("page", 0)
        txt = el.get("text", "") or ""
        n_txt = _normalize(txt)

        if typ in ("table", "heading", "list_item"):
            refined.append(el)
            continue

        buckets = per_page_sets.get(page, None)
        if not buckets or not n_txt:
            refined.append(el)
            continue

        # Prefer heading reclassification, then list item
        is_heading = any(_is_similar(n_txt, t) for t in buckets["title"])
        if is_heading:
            el["type"] = "heading"
            refined.append(el)
            continue

        is_list = any(_is_similar(n_txt, t) for t in buckets["list"])
        if is_list:
            el["type"] = "list_item"
            refined.append(el)
            continue

        # keep as paragraph
        refined.append(el)

    # Reindex after refinement (stable order preserved)
    for i, e in enumerate(refined):
        e["id"] = f"e_{i}"

    return refined


def extract_structural_elements(
    pdf_path: str,
    language: str,
    strategy: str = "fast",
    prefer_tables: bool = True,
    ensure_page_coverage: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extraction pipeline:
    1) Use Docling CLI with Granite-Docling-258M to convert the PDF into
       per-page HTML and parse block-level items (headings, paragraphs,
       list items, tables).
    2) Run Unstructured.partition_pdf to obtain per-page texts by type,
       and overlay the classification onto Docling segments to improve
       item typing (especially headings and list items).

    Returns a list of elements in the schema expected by downstream code:
      { id, text, type, page, html? }
    """
    print(f"\n--- PDF Processing (Docling Granite-Docling-258M + Unstructured overlay) ---")
    print(f"Docling converting: {pdf_path}")

    try:
        docling_elements = run_docling_granite(pdf_path, split_pages=True, show_layout=False)
        print(f"Docling elements parsed: {len(docling_elements)}")
    except Exception as exc:
        _log_err("Docling conversion failed", exc)
        # Hard fail: user requested Docling-only flow
        return []

    # Partition PDF for overlay classification
    try:
        raw_elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            infer_table_structure=bool(prefer_tables),
            languages=[language],
        )
        print(f"Unstructured partition elements: {len(raw_elements)}")
        final_elements = _overlay_types_with_partition(docling_elements, raw_elements)
        print(f"Final elements after overlay: {len(final_elements)}")
        return final_elements
    except Exception as exc:
        _log_err("Unstructured partition overlay failed", exc)
        # If overlay fails, still return Docling-only elements
        for i, e in enumerate(docling_elements):
            e["id"] = f"e_{i}"
        print(f"Returning Docling-only elements: {len(docling_elements)}")
        return docling_elements
