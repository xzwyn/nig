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