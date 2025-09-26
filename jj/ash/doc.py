import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Define the path to your local PDF file
pdf_file_path = "67g.pdf"

# Set up the document converter to use the Granite Docling model
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        ),
    }
)

print(f"Starting conversion for {pdf_file_path}...")

# Convert the document
doc = converter.convert(source=pdf_file_path).document

# Save the extracted content to a Markdown file
output_filename = "extracted_content.md"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())

print(f"✅ Success! Extracted content has been saved to {output_filename}")