import numpy as np
from typing import List, Tuple, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
