import streamlit as st
import pandas as pd
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, Header, Footer
from collections import Counter
import os
import io

# --- CORE LOGIC FUNCTIONS ---

def extract_sections_from_pdf(pdf_path, ocr_lang):
    """
    Extracts content from a PDF, segmenting it by titles.
    This version uses a robust coordinate-based filter to definitively ignore headers and footers.
    """
    try:
        elements = partition_pdf(
            pdf_path, 
            strategy="hi_res", 
            ocr_languages=ocr_lang,
            # We need coordinate data for positional filtering
            include_metadata=True 
        )
    except Exception as e:
        st.error(f"Error processing PDF '{os.path.basename(pdf_path)}': {e}")
        st.info("The 'hi_res' strategy requires Tesseract OCR to be installed and in your system's PATH.")
        return []

    sections = []
    current_heading = ""
    current_content = ""
    HEADING_CHAR_LIMIT = 150
    
    # --- Define Header/Footer positional thresholds ---
    # Any element starting in the top 10% or ending in the bottom 10% of a page will be ignored.
    HEADER_THRESHOLD_PERCENT = 0.10 
    FOOTER_THRESHOLD_PERCENT = 0.90

    for element in elements:
        # --- DEFINITIVE HEADER/FOOTER FILTERING BY POSITION ---
        # Get element coordinates and page height from metadata
        try:
            coords = element.metadata.coordinates
            page_height = element.metadata.page_height
            # The coordinate system's origin (0,0) is the top-left corner
            element_top_y = coords.points[0][1]
            element_bottom_y = coords.points[2][1]

            # Check if the element is within the header or footer zone
            if page_height and (element_top_y < page_height * HEADER_THRESHOLD_PERCENT or \
                                element_bottom_y > page_height * FOOTER_THRESHOLD_PERCENT):
                continue # Skip this element entirely
        except (AttributeError, TypeError, IndexError):
            # Fallback for elements without coordinate data
            pass

        # --- Secondary Filters (as a fallback) ---
        element_text = element.text.strip()
        if isinstance(element, (Header, Footer)):
            continue

        is_real_heading = isinstance(element, Title) and len(element_text) < HEADING_CHAR_LIMIT

        if is_real_heading:
            if current_heading:
                sections.append({
                    "heading": current_heading,
                    "content": current_content.strip()
                })
            
            current_heading = element_text
            current_content = ""
        else:
            if not current_heading:
                current_heading = "Initial Content (Before First Heading)"
            
            current_content += element.text + "\n"

    if current_heading:
        sections.append({
            "heading": current_heading,
            "content": current_content.strip()
        })
            
    return sections


def align_sections(english_sections, german_sections):
    """
    Aligns English and German sections into a pandas DataFrame.
    """
    aligned_data = []
    num_pairs = min(len(english_sections), len(german_sections))

    for i in range(num_pairs):
        eng_section = english_sections[i]
        ger_section = german_sections[i]
        
        eng_full_text = f"**{eng_section['heading']}**\n\n{eng_section['content']}"
        ger_full_text = f"**{ger_section['heading']}**\n\n{ger_section['content']}"
        
        aligned_data.append({
            "No": i + 1,
            "English": eng_full_text,
            "German": ger_full_text
        })
        
    if len(english_sections) > num_pairs:
        for i in range(num_pairs, len(english_sections)):
            eng_section = english_sections[i]
            eng_full_text = f"**{eng_section['heading']}**\n\n{eng_section['content']}"
            aligned_data.append({"No": i + 1, "English": eng_full_text, "German": ""})
            
    elif len(german_sections) > num_pairs:
        for i in range(num_pairs, len(german_sections)):
            ger_section = german_sections[i]
            ger_full_text = f"**{ger_section['heading']}**\n\n{ger_section['content']}"
            aligned_data.append({"No": i + 1, "English": "", "German": ger_full_text})
            
    return pd.DataFrame(aligned_data)

def to_excel(df):
    """
    Converts a DataFrame to an in-memory Excel file.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Aligned Content')
    processed_data = output.getvalue()
    return processed_data

def save_uploaded_file(uploaded_file, temp_dir="temp_files"):
    """
    Saves an uploaded file to a temporary directory to get a file path.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- STREAMLIT UI ---

def main():
    st.set_page_config(layout="wide", page_title="PDF Content Aligner")

    st.title("📄 PDF Content Aligner")
    st.markdown("""
    Upload your English and German PDFs. The tool will align content based on headings. 
    It automatically ignores text in the top and bottom 10% of each page to filter out headers and footers.
    """)

    col1, col2 = st.columns(2)
    with col1:
        english_pdf = st.file_uploader("Upload English PDF 🇬🇧", type="pdf")
    with col2:
        german_pdf = st.file_uploader("Upload German PDF 🇩🇪", type="pdf")

    if st.button("Process and Align Documents", type="primary"):
        if english_pdf and german_pdf:
            with st.spinner("Processing documents using hi_res model... This will take longer."):
                
                eng_pdf_path = save_uploaded_file(english_pdf)
                ger_pdf_path = save_uploaded_file(german_pdf)

                english_sections = extract_sections_from_pdf(eng_pdf_path, ocr_lang="eng")
                german_sections = extract_sections_from_pdf(ger_pdf_path, ocr_lang="deu")

                if english_sections and german_sections:
                    st.success(f"✅ Processing complete! Found {len(english_sections)} sections in English and {len(german_sections)} in German.")

                    aligned_df = align_sections(english_sections, german_sections)
                    
                    st.session_state.aligned_df = aligned_df
                    st.session_state.download_ready = True
                
                os.remove(eng_pdf_path)
                os.remove(ger_pdf_path)
        else:
            st.warning("⚠️ Please upload both PDF files to proceed.")

    if 'download_ready' in st.session_state and st.session_state.download_ready:
        st.header("Alignment Preview")
        st.dataframe(st.session_state.aligned_df)

        excel_data = to_excel(st.session_state.aligned_df)
        
        st.download_button(
            label="📥 Download Aligned Excel File",
            data=excel_data,
            file_name=f"aligned_content_{english_pdf.name.split('.')[0]}_{german_pdf.name.split('.')[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()