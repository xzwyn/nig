tMVbcVYCCRKjnRfFBpCfMctiLoqhRnHfGW
print("🚀 Step 1: Installing necessary libraries...")
!pip install -q docling torch

import os
from google.colab import files
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
print("✅ Libraries installed successfully.")
print("-" * 50)

print("\n⬆️ Step 2: Upload your PDF file")
uploaded = files.upload()

if not uploaded:
    raise Exception("No file was uploaded. Please run the cell again.")
else:
    pdf_filename = list(uploaded.keys())[0]
    print(f"\n✅ File '{pdf_filename}' uploaded successfully.")
    print("-" * 50)

def extract_text_from_pdf(file_path):
    print(f"\n🔍 Step 3: Starting text extraction from '{file_path}'...")
    print("This may take a few minutes as the model needs to be downloaded and run.")
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline),
        }
    )
    
    doc = converter.convert(source=file_path).document
    raw_markdown = doc.export_to_markdown()
    
    print("✅ Raw text extracted successfully.")
    print("-" * 50)
    return raw_markdown

def correct_paragraphs(content):
    print("\n✍️ Step 4: Correcting paragraph breaks...")
    paragraphs = content.split('\n\n')
    
    if not paragraphs:
        return ""

    corrected_paragraphs = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        current_paragraph = paragraphs[i].strip()
        previous_paragraph = corrected_paragraphs[-1].strip()

        if previous_paragraph and not previous_paragraph.startswith('#') and not previous_paragraph.endswith('.'):
            corrected_paragraphs[-1] = f"{previous_paragraph} {current_paragraph}"
        else:
            corrected_paragraphs.append(current_paragraph)
            
    corrected_text = '\n\n'.join(corrected_paragraphs)
    print("✅ Paragraphs corrected successfully.")
    print("-" * 50)
    return corrected_text

raw_md_content = extract_text_from_pdf(pdf_filename)
corrected_md_content = correct_paragraphs(raw_md_content)
output_filename = "corrected_output.md"

with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(corrected_md_content)

print(f"\n💾 Final corrected content saved to '{output_filename}'.")
print(f"\n⬇️ Step 5: Downloading your corrected file...")
files.download(output_filename)
print("\n🎉 All done!")
