// azure_client.py
# azure_client.py
import os
from typing import List, Dict, Any, Optional
import numpy as np
from openai import AzureOpenAI

_client: Optional[AzureOpenAI] = None
_cfg = {
    "endpoint": None,
    "api_key": None,
    "api_version": None,
    "chat_deployment": None,
    "embed_deployment": None,
}

def _load_env():
    _cfg["endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://mcphack-oai-01-swedencentral.openai.azure.com
    _cfg["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
    _cfg["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    _cfg["chat_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    _cfg["embed_deployment"] = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")  # optional

def _get_client() -> AzureOpenAI:
    global _client
    if _client is not None:
        return _client
    _load_env()
    if not _cfg["endpoint"] or not _cfg["api_key"]:
        raise RuntimeError("Azure OpenAI client is not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
    _client = AzureOpenAI(
        azure_endpoint=_cfg["endpoint"],
        api_key=_cfg["api_key"],
        api_version=_cfg["api_version"],
    )
    return _client

def chat(messages: List[Dict[str, Any]], temperature: float = 0.1, model: Optional[str] = None) -> str:
    client = _get_client()
    deployment = model or _cfg["chat_deployment"] or "gpt-4o"
    resp = client.chat.completions.create(
        model=deployment,  # must be your deployment name
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""

def embed(texts: List[str], model: Optional[str] = None, batch_size: int = 64) -> np.ndarray:
    client = _get_client()
    deployment = model or _cfg["embed_deployment"]
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is not set. Create an embeddings deployment or pass model=.")
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=deployment, input=chunk)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype=np.float32)

// agent.py
import json
from evaluator import evaluate_translation_pair, check_context_mismatch, evaluate_table_pair
from azure_client import chat  # NEW: use Azure OpenAI

# def _agent2_validate_finding(
#     eng_text: str,
#     ger_text: str,
#     error_type: str,
#     explanation: str,
#     model_name: str | None = None,
# ):
def _agent2_validate_finding(
    eng_text: str,
    ger_text: str,
    error_type: str,
    explanation: str,
    model_name: str | None = None,
):
    """
    Second-stage reviewer.  Confirms only truly fatal errors and rejects
    false positives.
    """
    prompt = f"""
## ROLE
**Senior Quality Reviewer** – you are the final gatekeeper of EN→DE
translation findings.

## TASK
Decide whether the finding delivered by Agent-1 must be *Confirmed* or
*Rejected*.

## INSTRUCTIONS
1. Eligible error_type values are **exactly**:
   • "Mistranslation"  
   • "Omission"

2. Confirm only when the evidence is unmistakable:
   • Mistranslation
       – number mismatch (digit or word)  
       – polarity flip / opposite meaning  
       – actor/role inversion  
   • Omission
       – English states an explicit count (“two”, “three”, “both” …) **or**
         lists concrete items, and at least one item is *truly* missing in
         German (not conveyed by paraphrase).

3. Reject when:
   • Difference is stylistic or synonymous.  
   • Proper names / document titles are rendered with an accepted German
     equivalent (e.g. “Nichtfinanzielle Erklärung”).  
   • Alleged omission is actually present via paraphrase.  

## OUTPUT ‑ JSON ONLY
json {{ "verdict" : "Confirm" | "Reject", "reasoning": "" }}

## MATERIAL TO REVIEW
English text:
\"\"\"{eng_text}\"\"\"

German text:
\"\"\"{ger_text}\"\"\"

Agent-1 proposed:
  error_type : {error_type}
  explanation: {explanation}

## YOUR RESPONSE
Return the JSON object only – no extra text.
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        )
        j0, j1 = content.find("{"), content.rfind("}") + 1
        verdict_json = json.loads(content[j0:j1])
        is_confirmed = verdict_json.get("verdict", "").lower() == "confirm"
        reasoning = verdict_json.get("reasoning", "")
        return is_confirmed, reasoning, content.strip()
    except (ValueError, json.JSONDecodeError) as exc:
        print("  - Agent-2 JSON parse error:", exc)
        return False, f"System error: {exc}", "{}"
    except Exception as exc:
        print("  - Agent-2 unexpected error:", exc)
        return False, "System error (non-parsing issue)", "{}"


def run_evaluation_pipeline(aligned_pairs: list, model_name: str = None):
    print("\n--- Agentic Evaluation Stage ---")

    for idx, (eng_elem, ger_elem) in enumerate(aligned_pairs):
        print(f"\n{'='*60}")
        eng_id = eng_elem['id'] if eng_elem else 'N/A'
        ger_id = ger_elem['id'] if ger_elem else 'N/A'
        print(f"Processing pair {idx}: ENG [{eng_id}] | GER [{ger_id}]")

        if eng_elem and not ger_elem:
            yield { "type": f"Omitted {eng_elem.get('type', 'Item')}", "english_text": eng_elem['text'], "german_text": "---", "suggestion": "Missing from German document", "page": eng_elem['page'] }
            continue
        if not eng_elem and ger_elem:
            yield { "type": f"Added {ger_elem.get('type', 'Item')}", "english_text": "---", "german_text": ger_elem['text'], "suggestion": "Extra item in German document", "page": ger_elem['page'] }
            continue

        if eng_elem and ger_elem:
            if eng_elem.get('type') == 'table':
                finding = evaluate_table_pair(eng_elem.get('html', ''), ger_elem.get('html', ''), model_name)
                print(f"  - Agent 1 Result (Table): {finding.get('error_type', 'None')}")
                if finding.get("error_type", "None") not in ["None", "System Error"]:
                    yield { "type": f"Table Error: {finding.get('error_type')}", "english_text": f"Table on page {eng_elem['page']}", "german_text": f"Table on page {ger_elem['page']}", "suggestion": finding.get("explanation"), "page": eng_elem['page'] }
                continue

            eng_text = eng_elem['text']
            ger_text = ger_elem['text']
            print(f"Evaluating: \"{eng_text[:60].strip()}...\"")

            finding = evaluate_translation_pair(eng_text, ger_text, model_name)
            error_type = finding.get("error_type", "None")
            print(f"Agent 1 Result: {error_type}")
            if error_type not in ["None", "System Error"]:
                print(f"Agent 1 JSON Response: {json.dumps(finding, indent=2)}")

                is_confirmed, reasoning, agent2_raw_response = _agent2_validate_finding(eng_text, ger_text, error_type, finding.get("explanation"), model_name)
                verdict_str = 'Accept' if is_confirmed else 'Reject'
                print(f"Agent 2 Response: {verdict_str}")
                print(f"Agent 2 JSON Response: {agent2_raw_response}")

                if is_confirmed:
                    yield { "type": error_type, "english_text": eng_text, "german_text": ger_text, "suggestion": finding.get("suggestion"), "page": eng_elem['page'], "original_phrase": finding.get("original_phrase"), "translated_phrase": finding.get("translated_phrase") }
            else:
                context_result = check_context_mismatch(eng_text, ger_text, model_name)
                context_match_verdict = context_result.get('context_match', 'Error')
                print(f"Agent 3 Context Match: {context_match_verdict}")
                print(f"Agent 3 JSON Response: {json.dumps(context_result, indent=2)}")
                if context_match_verdict.lower() == "no":
                    yield { "type": "Context Mismatch", "english_text": eng_text, "german_text": ger_text, "suggestion": context_result.get("explanation"), "page": eng_elem['page'] }

    print(f"\n--- Evaluation complete. ---")

// aligner.py
# aligner.py
import numpy as np
from typing import List, Tuple, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional Azure embeddings
try:
    from azure_client import embed as azure_embed
    AZURE_AVAILABLE = True
except Exception:
    AZURE_AVAILABLE = False


# ------------------------------------------------------------------ #
# Helper – pretty print a pair as soon as it is created
# ------------------------------------------------------------------ #
def _log_pair(idx: int,
              eng_item: Dict[str, Any] | None,
              ger_item: Dict[str, Any] | None):
    """
    Print a concise single-line summary of the pair that was just added
    to `final_pairs`.  Called from inside the alignment loop.
    """
    def snip(txt: str, n: int = 40):
        return (txt[: n - 1] + "…") if len(txt) > n else txt

    eng_id = eng_item.get("id") if eng_item else "—"
    ger_id = ger_item.get("id") if ger_item else "—"
    eng_pg = eng_item.get("page") if eng_item else "—"
    ger_pg = ger_item.get("page") if ger_item else "—"

    eng_txt = snip(eng_item.get("text", "")) if eng_item else ""
    ger_txt = snip(ger_item.get("text", "")) if ger_item else ""

    tag = (
        "[MATCH]" if eng_item and ger_item
        else "[OMIT ]" if eng_item and not ger_item
        else "[ADD  ]"
    )

    print(f"{tag} {idx:03d}: "
          f"ENG({eng_id}, p{eng_pg}) ↔ GER({ger_id}, p{ger_pg}) | "
          f"«{eng_txt}»  ↔  «{ger_txt}»")


# ------------------------------------------------------------------ #
# Embeddings / similarity back-end
# ------------------------------------------------------------------ #
def _compute_similarity_matrix(eng_texts: List[str], ger_texts: List[str]) -> Tuple[np.ndarray, str]:
    if AZURE_AVAILABLE:
        try:
            print("Embedding with Azure OpenAI (embeddings deployment)...")
            eng_emb = azure_embed(eng_texts)
            ger_emb = azure_embed(ger_texts)
            return cosine_similarity(eng_emb, ger_emb), "azure-embeddings"
        except Exception as exc:
            print(f"Azure embeddings failed, falling back to TF-IDF.  Reason: {exc}")

    print("Embedding with local TF-IDF fallback…")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    tfidf = vec.fit_transform(eng_texts + ger_texts)
    N = len(eng_texts)
    return cosine_similarity(tfidf[:N], tfidf[N:]), "tfidf"


# ------------------------------------------------------------------ #
# Public function
# ------------------------------------------------------------------ #
def align_documents(
    eng_elements: List[Dict[str, Any]],
    ger_elements: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any] | None, Dict[str, Any] | None]]:
    print("\n--- Document Alignment Stage (Azure-embeddings or TF-IDF) ---")

    item_types = {"heading", "paragraph", "table", "list_item"}
    eng_items = [e for e in eng_elements if e.get("type") in item_types]
    ger_items = [g for g in ger_elements if g.get("type") in item_types]

    if not eng_items or not ger_items:
        return []

    eng_texts = [e["text"] for e in eng_items]
    ger_texts = [g["text"] for g in ger_items]

    sim_matrix, backend = _compute_similarity_matrix(eng_texts, ger_texts)
    print(f"Similarity backend used: {backend}")

    # ------------------ Stage 1 : heading anchors ------------------- #
    print("\n--- Stage 1: Finding high-confidence heading anchors ---")
    eng_headings = {i: e for i, e in enumerate(eng_items) if e["type"] == "heading"}
    ger_headings = {i: g for i, g in enumerate(ger_items) if g["type"] == "heading"}

    heading_thr = 0.60 if backend == "azure-embeddings" else 0.15
    heading_anchors: Dict[int, int] = {}
    used_ger = set()

    for ei, eh in eng_headings.items():
        best_gi, best_sim = -1, heading_thr
        for gi, gh in ger_headings.items():
            if gi in used_ger:
                continue
            s = sim_matrix[ei, gi]
            if s > best_sim:
                best_sim, best_gi = s, gi
        if best_gi != -1:
            heading_anchors[ei] = best_gi
            used_ger.add(best_gi)
            print(f"[ANCHOR] ENG {ei} «{eh['text'][:30]}…»  ->  GER {best_gi} «{ger_headings[best_gi]['text'][:30]}…» (sim={best_sim:.3f})")

    # ------------------ Stage 2 : sequential walk ------------------- #
    print("\n--- Stage 2: Strict sequential alignment ---")
    final_pairs: List[Tuple[Any, Any]] = []
    eng_ptr = ger_ptr = 0
    pair_idx = 0

    while eng_ptr < len(eng_items) or ger_ptr < len(ger_items):

        # -- re-sync on heading anchor
        if eng_ptr in heading_anchors:
            target_ger = heading_anchors[eng_ptr]
            while ger_ptr < target_ger:
                final_pairs.append((None, ger_items[ger_ptr]))
                _log_pair(pair_idx, None, ger_items[ger_ptr])
                pair_idx += 1
                ger_ptr += 1

            final_pairs.append((eng_items[eng_ptr], ger_items[ger_ptr]))
            _log_pair(pair_idx, eng_items[eng_ptr], ger_items[ger_ptr])
            pair_idx += 1
            eng_ptr += 1
            ger_ptr += 1
            continue

        eng_item = eng_items[eng_ptr] if eng_ptr < len(eng_items) else None
        ger_item = ger_items[ger_ptr] if ger_ptr < len(ger_items) else None

        final_pairs.append((eng_item, ger_item))
        _log_pair(pair_idx, eng_item, ger_item)
        pair_idx += 1

        if eng_ptr < len(eng_items):
            eng_ptr += 1
        if ger_ptr < len(ger_items):
            ger_ptr += 1

    # ------------------ summary ------------------ #
    print("\n--- Alignment complete ---")
    print(f"Total pairs produced: {len(final_pairs)} "
          f"(ENG elements: {len(eng_items)}, GER elements: {len(ger_items)})\n")
    return final_pairs

// app.py
from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import os
from datetime import datetime

from pdf_processor import extract_structural_elements  # etc...
from aligner import align_documents
from agent import run_evaluation_pipeline
from ui_components import display_results, prepare_excel_download


# --- Page Configuration ---
st.set_page_config(page_title=" Translation Evaluator", layout="wide")

# --- UI ---
st.title(" Translation Evaluator")
st.markdown("Upload a source English PDF and its German translation to identify translation errors.")

# --- Session State Initialization ---
def init_session_state():
    st.session_state.setdefault('processing_started', False)
    st.session_state.setdefault('analysis_complete', False)
    st.session_state.setdefault('results', [])
    st.session_state.setdefault('aligned_pairs', [])

init_session_state()

# --- Helper Function ---
def save_uploaded_file(uploaded_file):
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Documents")
    english_pdf = st.file_uploader("Upload English PDF (Source)", type="pdf")
    german_pdf = st.file_uploader("Upload German PDF (Translation)", type="pdf")

    st.header("2. Start Analysis")
    if st.button("Run Evaluation", disabled=st.session_state.processing_started or not (english_pdf and german_pdf)):
        # Reset state for a new run
        st.session_state.processing_started = True
        st.session_state.analysis_complete = False
        st.session_state.results = []
        st.session_state.aligned_pairs = []
        
        with st.spinner("Step 1/2: Processing and aligning documents..."):
            eng_path = save_uploaded_file(english_pdf)
            ger_path = save_uploaded_file(german_pdf)
            eng_elements = extract_structural_elements(eng_path, language="eng", strategy="fast")
            ger_elements = extract_structural_elements(ger_path, language="deu", strategy="fast")
            st.session_state.aligned_pairs = align_documents(eng_elements, ger_elements)
        # Streamlit will automatically re-run to start the processing loop.

    # --- CHANGED: Download Button Logic ---
    st.header("3. Export Results")
    # This logic now safely handles the download button creation
    if st.session_state.processing_started or st.session_state.analysis_complete:
        # Check if there are any results to download
        if st.session_state.results:
            excel_data = prepare_excel_download(st.session_state.results)
            current_date = datetime.now().strftime("%Y-%m-%d")
            st.download_button(
                label="📥 Download Report as Excel",
                data=excel_data,
                file_name=f"Translation_Analysis_{current_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=False
            )
        else:
            # If no results yet, show a disabled button to prevent crashing
            st.download_button(
                label="📥 Download Report as Excel",
                data=b'', # Use empty bytes as a placeholder
                file_name=f"Translation_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=True
            )
    else:
        st.markdown("*(The download button will appear here once analysis begins.)*")
    # --- END OF CHANGE ---

# --- Main Display and Processing Area ---
st.header("Evaluation Results")

# This is the main processing block that runs after the button is pressed
if st.session_state.processing_started and not st.session_state.analysis_complete:
    total_pairs = len(st.session_state.aligned_pairs)
    st.info(f"Step 2/2: Evaluating {total_pairs} pairs...")
    progress_bar = st.progress(0)
    
    results_container = st.container()
    
    # Process all pairs using the generator and update the UI in a single block
    all_results = []
    for i, result in enumerate(run_evaluation_pipeline(st.session_state.aligned_pairs)):
        if result:
            all_results.append(result)
        
        # Update session state and UI components inside the loop
        st.session_state.results = all_results
        progress_bar.progress((i + 1) / total_pairs)
        
        with results_container:
            results_container.empty()
            display_results(st.session_state.results)

    # Mark as complete once the loop finishes
    st.session_state.analysis_complete = True
    st.session_state.processing_started = False # Allow for a new run
    st.rerun() # Rerun to update the button state and final messages

# This block displays the final state after processing is complete or before it starts
else:
    if st.session_state.analysis_complete:
        if not st.session_state.results:
            st.success("✅ Analysis complete. No significant errors were found.")
        else:
            display_results(st.session_state.results) # Display final results
    else:
        st.info("Upload documents and click 'Run Evaluation' to begin.")

// evaluator.py
import json
from azure_client import chat

def evaluate_translation_pair(eng_text: str, ger_text: str, model_name=None):
    prompt = f"""
## ROLE
You are the **Primary Translation Auditor** for EN→DE corporate reports.

## TASK
Detect exactly two kinds of *fatal* errors and output ONE JSON object.

## ERROR TYPES YOU MAY REPORT
1. **Mistranslation**  
   • Wrong numeric value (digit or number word)  
   • Polarity flip / opposite meaning (e.g. *comprehensive → unvollständig*)  
   • Change of actor or role (passive “was informed” → active “initiated”)

2. **Omission**  
   The English text announces a specific count (“two”, “three”, “both” …)  
   or lists concrete items, and ≥1 of those items is missing in German.

Everything else—stylistic differences, synonyms, accepted German titles
(“Nichtfinanzielle Erklärung”, “Erklärung zur Unternehmensführung” …)—
is **not** an error.

## JSON OUTPUT SCHEMA
json {{ "error_type" : "Mistranslation" | "Omission" | "None", "original_phrase" : "", "translated_phrase": "", "explanation" : "<≤40 words>", "suggestion" : "" }}

## POSITIVE EXAMPLES
### 1 · Mistranslation (number)
EN “…held **five** meetings.”  
DE “…hielt **acht** Sitzungen.”  
→ error_type “Mistranslation”, original “five”, translated “acht”

### 2 · Omission
EN “Three focus areas: A, B, C.”  
DE “Drei Schwerpunkte: A und C.”  
→ error_type “Omission”, original “B”, translated ""

## NEGATIVE EXAMPLE (should be *None*)
EN “Non-Financial Statement”  
DE “Nichtfinanzielle Erklärung”

## TEXTS TO AUDIT
<Original English>
{eng_text}
</Original English>

<German Translation>
{ger_text}
</German Translation>

## YOUR RESPONSE
Return the JSON object only—no extra text, no markdown.
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        if j0 != -1 and j1 != -1:
            return json.loads(content[j0:j1])
        return {"error_type": "System Error",
                "explanation": "No JSON object in LLM reply."}
    except Exception as exc:
        print("evaluate_translation_pair →", exc)
        return {"error_type": "System Error", "explanation": str(exc)}


def check_context_mismatch(eng_text: str, ger_text: str, model_name: str = None):
    prompt = f"""
ROLE: Narrative-Integrity Analyst

Goal: Decide if the German text tells a **different story** from the
English.  “Different” means a change in
• WHO does WHAT to WHOM
• factual outcome or direction of action
• polarity (e.g. “comprehensive” ↔ “unvollständig”)

Ignore style, word order, or minor re-phrasing.

Respond with JSON:

{{
  "context_match": "Yes" | "No",
  "explanation":  "<one concise sentence>"
}}

Examples
--------
1) Role reversal (should be No)
EN  Further, the committee *was informed* by the Board …
DE  Darüber hinaus *leitete der Ausschuss eine Untersuchung ein* …
→ roles flipped ⇒ "No"

2) Identical meaning (Yes)
EN  Declaration of Conformity with the German Corporate Governance Code
DE  Entsprechenserklärung zum Deutschen Corporate Governance Kodex
→ "Yes"

Analyse the following text pair and respond with the JSON only.

<Original_English>
{eng_text}
</Original_English>

<German_Translation>
{ger_text}
</German_Translation>
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        return json.loads(content[j0:j1])
    except Exception as exc:
        return {"context_match": "Error", "explanation": str(exc)}


def evaluate_table_pair(eng_html: str, ger_html: str, model_name: str = None):
    prompt = f"""
You are a data auditor comparing two HTML tables.  Report ONLY critical
differences in JSON:

{{"error_type": "None" | "Structural_Mismatch" | "Data_Mismatch" |
  "Header_Error", "explanation": "..." }}

- English Table: {eng_html}
- German  Table: {ger_html}
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        return json.loads(content[j0:j1])
    except Exception as exc:
        return {"error_type": "System Error", "explanation": str(exc)}

// pdf_processor.py
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

// requirements.txt
streamlit

pymupdf
pytesseract
Pillow

sentence-transformers
numpy
scikit-learn
scipy
dotenv

unstructured
unstructured-inference
unstructured-client
unstructured[pdf]
pandas
openpyxl

openai>=1.40.0

// ui_components.py
# ui_components.py

import streamlit as st
import pandas as pd
import io

def display_results(results_list: list):
    """
    Renders the list of findings, now with a dedicated section for
    displaying granular phrase-level errors.
    """
    if not results_list:
        st.info("No significant translation errors were found in the analysis.")
        return

    st.subheader(f"Found {len(results_list)} noteworthy items")
    results_list.sort(key=lambda x: x.get('page', 0))

    for result in results_list:
        error_type = result.get('type', 'Info')
        
        with st.container(border=True):
            st.markdown(f"**Page:** `{result.get('page', 'N/A')}` | **Type:** `{error_type}`")
            
            # --- NEW: Granular Display for Phrase-Level Errors ---
            original_phrase = result.get("original_phrase")
            translated_phrase = result.get("translated_phrase")

            if original_phrase and translated_phrase:
                st.markdown("##### 🔍 Error Focus")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("Original English Phrase:")
                    st.error(f"'{original_phrase}'")
                with col2:
                    st.markdown("Translated German Phrase:")
                    st.warning(f"'{translated_phrase}'")
                st.divider()

            # Display full text for context, unless it's a table error
            if "Table Error" not in error_type:
                st.markdown("##### Full Text Context")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"> {result['english_text']}")
                with col2:
                    st.markdown(f"> {result['german_text']}")

            st.markdown(f"**💡 Suggestion:** {result['suggestion']}")


def prepare_excel_download(results_list: list):
    """
    Converts the list of results into an Excel file in memory for downloading.
    Now includes the specific phrases.
    """
    if not results_list:
        return None

    df_data = {
        "Page": [r.get('page', 'N/A') for r in results_list],
        "Error Type": [r.get('type') for r in results_list],
        "Original Phrase": [r.get('original_phrase', 'N/A') for r in results_list],
        "Translated Phrase": [r.get('translated_phrase', 'N/A') for r in results_list],
        "Suggestion": [r.get('suggestion') for r in results_list],
        "Full English Source": [r.get('english_text') for r in results_list],
        "Full German Translation": [r.get('german_text') for r in results_list]
    }
    
    df = pd.DataFrame(df_data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Translation_Analysis')
    
    processed_data = output.getvalue()
    return processed_data

