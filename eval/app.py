import streamlit as st
import ollama
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Translation Evaluator 🇩🇪",
    page_icon="🤖",
    layout="wide"
)

# --- LLM Call Function ---
def get_llm_response(prompt_text):
    """Function to get response from the Ollama model."""
    try:
        response = ollama.chat(
            model='aya:latest',
            messages=[{'role': 'user', 'content': prompt_text}],
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
        return None

# --- Evaluation Functions ---

def check_omissions(source_lines, full_target_text):
    """Check for omitted lines using the LLM."""
    omissions = []
    progress_bar = st.progress(0, text="Checking for omissions...")
    for i, line in enumerate(source_lines):
        if not line.strip():  # Skip empty lines
            continue
        
        prompt = f"""
        Is the essential meaning of the following English sentence present anywhere in the German text provided?
        Respond with only 'Yes' or 'No'. No explanation needed.

        English Sentence: "{line}"

        German Text:
        ---
        {full_target_text}
        ---
        """
        response = get_llm_response(prompt)
        if response and 'no' in response.lower():
            omissions.append({"Line": i + 1, "English Text": line})
        
        progress_bar.progress((i + 1) / len(source_lines), text=f"Checking line {i+1}/{len(source_lines)} for omission...")
    progress_bar.empty()
    return omissions

def check_mistranslations(source_lines, target_lines):
    """Check for mistranslations line by line."""
    mistranslations = []
    line_count = min(len(source_lines), len(target_lines))
    progress_bar = st.progress(0, text="Checking for mistranslations...")

    for i in range(line_count):
        source_line = source_lines[i]
        target_line = target_lines[i]

        if not source_line.strip() and not target_line.strip():
            continue

        prompt = f"""
        You are an expert German-English translation evaluator.
        Evaluate if the German sentence is an accurate translation of the English sentence.
        
        Your response must be in this format:
        Rating: [Correct, Minor Mistranslation, Major Mistranslation] | Explanation: [Briefly explain your reasoning.]

        English: "{source_line}"
        German: "{target_line}"
        """
        response = get_llm_response(prompt)
        if response and 'mistranslation' in response.lower():
            mistranslations.append({
                "Line": i + 1,
                "English Text": source_line,
                "German Translation": target_line,
                "Evaluation": response
            })
            
        progress_bar.progress((i + 1) / line_count, text=f"Evaluating line {i+1}/{line_count} for mistranslation...")
    progress_bar.empty()
    return mistranslations


# --- Streamlit UI ---

st.title("Translation Evaluator")

col1, col2 = st.columns(2)

with col1:
    st.header("Source Text (English)")
    source_text = st.text_area("Enter the original English text", height=300, key="source")

with col2:
    st.header("Translated Text (German)")
    target_text = st.text_area("Enter the German translation", height=300, key="target")

if st.button("Evaluate Translation", type="primary", use_container_width=True):
    if not source_text or not target_text:
        st.warning("Please enter both the source and target texts.")
    else:
        source_lines = source_text.strip().split('\n')
        target_lines = target_text.strip().split('\n')

        with st.spinner("Evaluation in progress."):
            # --- Omission Check ---
            st.subheader("⚠️ Omission Check Results")
            omitted_lines = check_omissions(source_lines, target_text)
            if not omitted_lines:
                st.success("✅ No line omissions detected.")
            else:
                st.error(f"Found {len(omitted_lines)} potentially omitted lines.")
                df_omissions = pd.DataFrame(omitted_lines)
                st.table(df_omissions)

            # --- Mistranslation Check ---
            st.subheader("🔍 Mistranslation Check Results")
            mistranslated_lines = check_mistranslations(source_lines, target_lines)
            if not mistranslated_lines:
                st.success("✅ No mistranslations detected in corresponding lines.")
            else:
                st.error(f"Found {len(mistranslated_lines)} potential mistranslations.")
                df_mistranslations = pd.DataFrame(mistranslated_lines)
                st.dataframe(df_mistranslations)

        st.info("💡 Note: The line count for source and target text should ideally match for the mistranslation check.")