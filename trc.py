# ==============================================================================
# Step 1: Install necessary packages for Kaggle
# ==============================================================================
!pip install -q unstructured sentence-transformers
!pip install -q "unstructured[pdf]"
!pip install -q nltk pandas

import os
import re
import json
import nltk
import pandas as pd
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer, util

# Download the sentence tokenizer model from NLTK
nltk.download('punkt', quiet=True)

def clean_text_elements(elements):
    """
    Cleans a list of elements extracted by 'unstructured'.
    Removes headers, footers, and other identified noise based on our analysis.
    """
    cleaned_text = ""
    # Regex to find potential page numbers or very short fragments
    page_num_pattern = re.compile(r'^\s*\d{1,3}\s*$')
    # Regex for table of contents style lines
    toc_pattern = re.compile(r'\.{5,}')

    for el in elements:
        text = el.text.strip()
        # --- Filtering Logic ---
        # 1. Skip if it's a known header/footer phrase
        if "Mercedes-Benz Group AG Annual Report" in text:
            continue
        # 2. Skip if it looks like a table of contents entry
        if toc_pattern.search(text):
            continue
        # 3. Skip if it's likely a standalone page number
        if page_num_pattern.match(text):
            continue
        # 4. Skip if the fragment is too short to be a useful sentence
        if len(text.split()) < 5:
            continue
        # If it passes all checks, add it to our clean text block
        cleaned_text += text + " "

    # Final cleanup of extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def extract_and_clean_pdf(pdf_path, languages):
    """
    Uses 'unstructured' to extract elements, then applies our custom cleaning function.
    """
    print(f"📄 Processing '{pdf_path}' with unstructured using language(s): {languages}...")
    # Use the "hi_res" strategy for best results with complex layouts
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        languages=languages,  # Changed from tesseract_language to languages
    )
    print(f"🧹 Cleaning extracted text for '{pdf_path}'...")
    cleaned_text = clean_text_elements(elements)
    return cleaned_text

def create_sentence_pairs(text_en, text_de, model, similarity_threshold=0.85):
    """Aligns sentences using semantic similarity."""
    sentences_en = nltk.sent_tokenize(text_en)
    sentences_de = nltk.sent_tokenize(text_de)

    print(f"\nFound {len(sentences_en)} English sentences and {len(sentences_de)} German sentences after cleaning.")

    print("🧠 Generating embeddings... (This is the most time-consuming step)")
    embeddings_en = model.encode(sentences_en, convert_to_tensor=True, show_progress_bar=True)
    embeddings_de = model.encode(sentences_de, convert_to_tensor=True, show_progress_bar=True)

    print("🤝 Calculating similarities and finding best pairs...")
    cosine_scores = util.cos_sim(embeddings_en, embeddings_de)

    aligned_pairs = []
    for i in range(len(sentences_en)):
        best_match_idx = cosine_scores[i].argmax()
        if cosine_scores[i][best_match_idx] > similarity_threshold:
            aligned_pairs.append({
                'english': sentences_en[i],
                'german': sentences_de[best_match_idx],
                'similarity': cosine_scores[i][best_match_idx].item()
            })
    return aligned_pairs

def format_for_finetuning(pairs):
    """Formats the aligned pairs into JSONL for fine-tuning."""
    finetuning_data = []
    for pair in pairs:
        # English -> German instruction
        finetuning_data.append(json.dumps({
            "instruction": "Translate the following English financial text to German with formal accuracy.",
            "input": pair['english'],
            "output": pair['german']
        }))
        # German -> English instruction
        finetuning_data.append(json.dumps({
            "instruction": "Translate the following German financial text to English with formal accuracy.",
            "input": pair['german'],
            "output": pair['english']
        }))
    return finetuning_data

# ==============================================================================
# Step 3: Upload your files and run the main process
# ==============================================================================
# In Kaggle, upload your PDFs to the input section or use the file paths below.
# For Kaggle, the paths will typically be in /kaggle/input/ or /kaggle/working/
english_pdf_path = '/content/en_2023.pdf'  # Update with your actual dataset name
german_pdf_path = '/content/en_2023.pdf'   # Update with your actual dataset name

# Alternative: If you uploaded files directly to the working directory
# english_pdf_path = '/kaggle/working/en_2024.pdf'
# german_pdf_path = '/kaggle/working/de_2024.pdf'

if not os.path.exists(english_pdf_path) or not os.path.exists(german_pdf_path):
    print("\n🛑 ERROR: Please upload 'en_2024.pdf' and 'de_2024.pdf' to the Colab session files.")
else:
    print("\n🚀 Starting dataset creation process...")
    # 1. Load the pre-trained multilingual model
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    # 2. Extract and clean text using our new functions with correct language parameters
    english_text = extract_and_clean_pdf(english_pdf_path, languages=["eng"])  # Changed parameter
    german_text = extract_and_clean_pdf(german_pdf_path, languages=["deu"])    # Changed parameter

    # 3. Create aligned sentence pairs
    aligned_pairs = create_sentence_pairs(english_text, german_text, model, similarity_threshold=0.85)
    print(f"\n🎉 Found {len(aligned_pairs)} high-quality translation pairs!")

    # 4. Save a CSV file for review (optional but recommended)
    df = pd.DataFrame(aligned_pairs)
    df.to_csv('/content/aligned_financial_sentences_cleaned.csv', index=False)
    print("\nSaved a reviewable CSV file: '/content/aligned_financial_sentences_cleaned.csv'")

    # 5. Format the data for fine-tuning and save as a JSONL file
    finetuning_jsonl = format_for_finetuning(aligned_pairs)
    output_filename = '/content/financial_translation_dataset.jsonl'
    with open(output_filename, 'w', encoding='utf-8') as f:
        for line in finetuning_jsonl:
            f.write(line + '\n')

    print(f"\n✅ Success! Your fine-tuning dataset with {len(finetuning_jsonl)} bidirectional instructions is ready.")
    print(f"File saved as: '{output_filename}'")
    print("\n📁 Output files are saved in /kaggle/working/ - you can download them or commit the notebook to save them.")
