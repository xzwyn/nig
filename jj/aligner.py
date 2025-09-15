from sentence_transformers import SentenceTransformer, util
import torch

def align_documents(eng_elements: list, ger_elements: list):
    print("\n--- Document Alignment Stage (Strict Sequential with Heading Re-sync) ---")

    item_types_to_process = ['heading', 'paragraph', 'table', 'list_item']
    eng_items = [el for el in eng_elements if el['type'] in item_types_to_process]
    ger_items = [el for el in ger_elements if el['type'] in item_types_to_process]

    if not eng_items or not ger_items:
        return []

    eng_texts = [item['text'] for item in eng_items]
    ger_texts = [item['text'] for item in ger_items]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('sentence-transformers/LaBSE', device=device)
    eng_embeddings = model.encode(eng_texts, convert_to_tensor=True, show_progress_bar=True)
    ger_embeddings = model.encode(ger_texts, convert_to_tensor=True, show_progress_bar=True)
    similarity_matrix = util.cos_sim(eng_embeddings, ger_embeddings).cpu().numpy()

    print("\n--- Stage 1: Finding High-Confidence Heading Anchors ---")
    eng_headings = {i: item for i, item in enumerate(eng_items) if item['type'] == 'heading'}
    ger_headings = {i: item for i, item in enumerate(ger_items) if item['type'] == 'heading'}
    
    heading_anchors = {} # Maps English heading index -> German heading index
    used_ger_headings = set()
    for eng_idx, eng_h in eng_headings.items():
        best_ger_idx = -1

        best_sim = 0.75  
        for ger_idx, ger_h in ger_headings.items():
            if ger_idx in used_ger_headings:
                continue
            sim = similarity_matrix[eng_idx][ger_idx]
            if sim > best_sim:
                best_sim = sim
                best_ger_idx = ger_idx
        if best_ger_idx != -1:
            print(f"[ANCHOR ✔️] Locking ENG Heading '{eng_h['text'][:30]}...' to GER Heading '{ger_headings[best_ger_idx]['text'][:30]}...'")
            heading_anchors[eng_idx] = best_ger_idx
            used_ger_headings.add(best_ger_idx)

    print("\n--- Stage 2: Performing Strict Sequential Alignment ---")
    final_pairs = []
    eng_ptr, ger_ptr = 0, 0

    while eng_ptr < len(eng_items) or ger_ptr < len(ger_items):
        
        if eng_ptr in heading_anchors:
            target_ger_ptr = heading_anchors[eng_ptr]
            
            while ger_ptr < target_ger_ptr:
                final_pairs.append((None, ger_items[ger_ptr]))
                ger_ptr += 1
            
            if eng_ptr < len(eng_items) and ger_ptr < len(ger_items):
                final_pairs.append((eng_items[eng_ptr], ger_items[ger_ptr]))
                eng_ptr += 1
                ger_ptr += 1
            continue

        eng_item = eng_items[eng_ptr] if eng_ptr < len(eng_items) else None
        ger_item = ger_items[ger_ptr] if ger_ptr < len(ger_items) else None

        if eng_item is not None or ger_item is not None:
            final_pairs.append((eng_item, ger_item))
        
        if eng_ptr < len(eng_items):
            eng_ptr += 1
        if ger_ptr < len(ger_items):
            ger_ptr += 1

    print("\n" + "="*80)
    print("--- Final Alignment Verification ---")
    for eng, ger in final_pairs:
        if eng and ger:
            print(f"[MATCHED] ENG {eng['id']} <-> GER {ger['id']}")
        elif eng and not ger:
            print(f"[OMITTED] ENG {eng['id']}")
        elif not eng and ger:
            print(f"[ADDED]   GER {ger['id']}")
    print("="*80 + "\n")
    
    return final_pairs