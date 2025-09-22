# docling_client.py
import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup


"""
Docling + Granite-Docling-258M integration.

This module runs the Docling CLI with the Granite-Docling VLM backend to
convert a PDF into per-page HTML, then parses that HTML into the app's
element schema:
  {
    "id": "e_<int>",
    "text": "<string>",                 # empty for tables
    "type": "heading"|"paragraph"|"list_item"|"table",
    "page": <int>,
    "html": "<table html>"              # only for type == "table"
  }
"""


def _safe_run(cmd: list, cwd: Optional[str] = None, timeout: Optional[int] = None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc


def _page_from_filename(name: str) -> int:
    # Extract page number from names like: page_001.html, 001.html, page-12.html, etc.
    m = re.search(r'(\d+)', name)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _iter_block_elements(soup: BeautifulSoup):
    """
    Iterate over block-level elements in reading order:
    - headings: h1..h6
    - paragraphs: p
    - list items: li
    - tables: table
    We yield the <table> as a whole and do not descend into it to avoid duplicates.
    """
    body = soup.body or soup
    for el in body.descendants:
        if not hasattr(el, "name"):
            continue
        name = (el.name or "").lower()
        if name == "table":
            # Yield the table element and skip traversing its children
            yield el
            for _ in el.descendants:
                pass
            continue
        if name in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li"):
            yield el


def _classify(el) -> str:
    name = (el.name or "").lower()
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return "heading"
    if name == "li":
        return "list_item"
    if name == "table":
        return "table"
    return "paragraph"


def _clean_text(txt: str) -> str:
    txt = re.sub(r'\r', ' ', txt)
    # collapse multiple blank lines
    txt = re.sub(r'\n\s*\n+', '\n', txt)
    # collapse spaces/tabs
    txt = re.sub(r'[ \t]+', ' ', txt)
    return txt.strip()


def parse_docling_page_html(html_path: str, page_number: int) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    idx = 0
    seen_tables = set()

    for el in _iter_block_elements(soup):
        typ = _classify(el)
        if typ == "table":
            tbl_html = el.prettify()
            hsh = hash(tbl_html)
            if hsh in seen_tables:
                continue
            seen_tables.add(hsh)
            elements.append({
                "id": f"e_{idx}",
                "text": "",
                "type": "table",
                "page": page_number,
                "html": tbl_html,
            })
            idx += 1
            continue

        txt = _clean_text(el.get_text(separator="\n"))
        if not txt:
            continue
        elements.append({
            "id": f"e_{idx}",
            "text": txt,
            "type": typ,
            "page": page_number,
        })
        idx += 1

    return elements


def run_docling_granite(
    pdf_path: str,
    split_pages: bool = True,
    show_layout: bool = False,
    timeout_sec: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Runs the Docling CLI with Granite-Docling VLM backend.

    Command (split pages):
      docling --to html_split_page --pipeline vlm --vlm-model granite_docling <abs_pdf_path>

    Uses a temporary working directory as the CLI's CWD so all artifacts
    land there. IMPORTANT: we pass an absolute input path so it resolves
    regardless of CWD changes (fixes 'file does not exist' errors).
    """
    # Make input absolute BEFORE changing CWD
    input_pdf_abs = os.path.abspath(os.path.normpath(pdf_path))
    if not os.path.isfile(input_pdf_abs):
        raise FileNotFoundError(input_pdf_abs)

    out_dir = tempfile.mkdtemp(prefix="docling_out_")
    try:
        cmd = ["docling"]
        if split_pages:
            cmd += ["--to", "html_split_page"]
        else:
            cmd += ["--to", "html"]
        if show_layout:
            cmd += ["--show-layout"]
        cmd += ["--pipeline", "vlm", "--vlm-model", "granite_docling", input_pdf_abs]

        # Run in the temp output directory to collect artifacts neatly
        _safe_run(cmd, cwd=out_dir, timeout=timeout_sec)

        # Collect generated HTML files
        html_files: List[str] = []
        for root, _, files in os.walk(out_dir):
            for name in files:
                if name.lower().endswith(".html"):
                    html_files.append(os.path.join(root, name))

        # If split pages, prefer files that look like per-page outputs
        if split_pages:
            per_page = [p for p in html_files if re.search(r'(page[_\-]?\d+|^\d+)\.html$', os.path.basename(p), re.I)]
            if per_page:
                html_files = per_page

        # Sort by inferred page number
        try:
            html_files.sort(key=lambda p: _page_from_filename(os.path.basename(p)))
        except Exception:
            html_files.sort()

        elements: List[Dict[str, Any]] = []
        for hf in html_files:
            page = _page_from_filename(os.path.basename(hf))
            elements.extend(parse_docling_page_html(hf, page))

        # Reindex globally
        for i, el in enumerate(elements):
            el["id"] = f"e_{i}"

        return elements
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
