# app.py
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

            # Docling Granite-Docling-258M extraction + Unstructured overlay classification
            eng_elements = extract_structural_elements(eng_path, language="eng", strategy="fast")
            ger_elements = extract_structural_elements(ger_path, language="deu", strategy="fast")

            st.session_state.aligned_pairs = align_documents(eng_elements, ger_elements)
        # Streamlit will automatically re-run to start the processing loop.

    # --- Download Button Logic ---
    st.header("3. Export Results")
    if st.session_state.processing_started or st.session_state.analysis_complete:
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
            st.download_button(
                label="📥 Download Report as Excel",
                data=b'',
                file_name=f"Translation_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=True
            )
    else:
        st.markdown("*(The download button will appear here once analysis begins.)*")

# --- Main Display and Processing Area ---
st.header("Evaluation Results")

if st.session_state.processing_started and not st.session_state.analysis_complete:
    total_pairs = len(st.session_state.aligned_pairs)
    st.info(f"Step 2/2: Evaluating {total_pairs} pairs...")
    progress_bar = st.progress(0)

    results_container = st.container()

    all_results = []
    for i, result in enumerate(run_evaluation_pipeline(st.session_state.aligned_pairs)):
        if result:
            all_results.append(result)

        st.session_state.results = all_results
        progress_bar.progress((i + 1) / total_pairs)

        with results_container:
            results_container.empty()
            display_results(st.session_state.results)

    st.session_state.analysis_complete = True
    st.session_state.processing_started = False
    st.rerun()

else:
    if st.session_state.analysis_complete:
        if not st.session_state.results:
            st.success("✅ Analysis complete. No significant errors were found.")
        else:
            display_results(st.session_state.results)
    else:
        st.info("Upload documents and click 'Run Evaluation' to begin.")
