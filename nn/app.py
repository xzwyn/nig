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