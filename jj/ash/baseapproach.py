from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# ---------------------------
# 1. Extract text paragraphs
# ---------------------------
def extract_paragraphs(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            # Split by double newlines or line breaks
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            text.extend(paragraphs)
    return text

eng_path = "test_1_e.pdf"
ger_path = "test_1_g.pdf"

eng_paragraphs = extract_paragraphs(eng_path)
ger_paragraphs = extract_paragraphs(ger_path)

print(f"Extracted {len(eng_paragraphs)} English paragraphs and {len(ger_paragraphs)} German paragraphs.")

# ---------------------------
# 2. Load multilingual model
# ---------------------------
# "paraphrase-multilingual-MiniLM-L12-v2" is a good balance of speed and quality
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ---------------------------
# 3. Encode both sets
# ---------------------------
eng_embeddings = model.encode(eng_paragraphs, convert_to_tensor=True)
ger_embeddings = model.encode(ger_paragraphs, convert_to_tensor=True)

# ---------------------------
# 4. Find best matches
# ---------------------------
results = []
for i, eng_emb in enumerate(eng_embeddings):
    # Compute cosine similarity with all German paragraphs
    cosine_scores = util.cos_sim(eng_emb, ger_embeddings)[0]
    best_match_id = int(cosine_scores.argmax())
    best_score = float(cosine_scores[best_match_id])

    results.append({
        "English Paragraph": eng_paragraphs[i][:200] + ("..." if len(eng_paragraphs[i]) > 200 else ""),
        "German Match": ger_paragraphs[best_match_id][:200] + ("..." if len(ger_paragraphs[best_match_id]) > 200 else ""),
        "Similarity Score": round(best_score, 3),
        "Match Index": best_match_id
    })

# ---------------------------
# 5. Convert to DataFrame
# ---------------------------
df = pd.DataFrame(results)

# Sort by similarity ascending (low = potential mistranslation/omission)
df_sorted = df.sort_values(by="Similarity Score").reset_index(drop=True)

# Save results
df_sorted.to_excel("translation_alignment_results.xlsx", index=False)

print("Done! Check 'translation_alignment_results.xlsx' for flagged mismatches.")
