import ollama
import json
import re
def evaluate_translation_pair(eng_text: str, ger_text: str, model_name: str = 'mistral:latest'):
    prompt = f"""
# ROLE
You are a German-English Translation Auditor focused on finding only the most critical, undeniable errors.

# TASK
Your task is to find fatal errors in the German translation. You will ignore any minor issues, stylistic differences, or ambiguous phrasing.

# STRICT RULES OF ENGAGEMENT
1.  **Fatal Errors Only**: You will ONLY flag errors that are factually and objectively wrong.
2.  **Ignore Ambiguity**: Do not flag issues of style, tone, or potential misunderstanding. If an error requires deep interpretation to understand, it is NOT a critical error.

# CRITICAL ERROR CATEGORIES
1.  **Factual Mismatch**: The translation changes a specific fact, especially numbers written as words (e.g., 'five' vs. 'acht'). This is a critical Mistranslation.
2.  **Opposite Meaning**: The translation uses a word that means the opposite of the original (e.g., 'comprehensive' vs. 'incomplete'). This is a critical Mistranslation.
3.  **Structural Omission (Rare)**: This error applies ONLY when a text explicitly states a number of items in a list (e.g., "the three components are...") and the translation fails to include all of them. This is a structural failure.

# INSTRUCTIONS
- If you find an error from the categories above, report it in JSON format.
- If there are no such fatal errors, you MUST return `{{"error_type": "None"}}`.

# EXAMPLES
---
**Example 1: Factual Mismatch (Number Word)**
<Original_English>
The committee held five meetings.
</Original_English>
<German_Translation>
Der Ausschuss hielt acht Sitzungen ab.
</German_Translation>
- Desired JSON Output:
{{
  "error_type": "Mistranslation",
  "original_phrase": "five",
  "translated_phrase": "acht",
  "explanation": "A number written as a word was factually changed from 'five' to 'acht' (eight).",
  "suggestion": "fünf"
}}
---
**Example 2: Structural Omission (The ONLY type of omission to report)**
<Original_English>
The three focus areas were: review of Alpha, effectiveness of IT outsourcing, and a re-test of controls.
</Original_English>
<German_Translation>
Die drei Schwerpunkte waren: Überprüfung von Alpha und eine Wiederholung der Kontrollen.
</German_Translation>
- Desired JSON Output:
{{
  "error_type": "Omission",
  "original_phrase": "effectiveness of IT outsourcing",
  "explanation": "A key item was dropped from an explicitly enumerated list of three focus areas, breaking the document's structure.",
  "suggestion": "Überprüfung von Alpha, Wirksamkeit des IT-Outsourcings und eine Wiederholung der Kontrollen."
}}
---

# YOUR TASK
Analyze the texts for fatal errors based on the strict rules above.

<Original_English>
{eng_text}
</Original_English>

<German_Translation>
{ger_text}
</German_Translation>
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        content = response['message']['content'].strip()
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        else:
            return {"error_type": "System Error", "explanation": "LLM did not return a valid JSON object."}
    except Exception as e:
        print(f"An error occurred while contacting Ollama: {e}")
        return {"error_type": "System Error", "explanation": str(e)}

def check_context_mismatch(eng_text: str, ger_text: str, model_name: str = 'mistral:latest'):
    prompt = f"""
# ROLE
You are a Narrative Integrity Analyst.

# TASK
Your task is to determine if the German text describes a fundamentally different event, narrative, or outcome compared to the English text. Focus on the "story" being told, not just the grammar.

# INSTRUCTIONS
1.  Read both texts and understand the situation they describe.
2.  A **Context Mismatch** occurs if the core narrative changes in a significant way. For example, if the role of a subject changes from being a passive recipient of information to an active initiator of a new event.
3.  Ignore minor differences in wording if the overall story is the same. **Do not flag differences in simple titles or headings that have the same meaning.**
4.  Provide your answer in a strict JSON format.

# EXAMPLES
---
**Example 1: CONTEXT MISMATCH (Change in Narrative and Role)**
<Original_English>
Furthermore, the committee was regularly informed by the Board about the status of the measures.
</Original_English>
<German_Translation>
Darüber hinaus leitete der Ausschuss eine neue interne Untersuchung der Maßnahmen des Vorstands ein.
</German_Translation>
- Desired JSON Output:
{{
  "context_match": "No",
  "explanation": "The narrative has fundamentally changed. The committee's role shifted from being a passive recipient of information ('was informed') to being an active initiator of a new event ('launched an investigation')."
}}
---
**Example 2: Context Matches (Same Title)**
<Original_English>
Declaration of Conformity with the German Corporate Governance Code
</Original_English>
<German_Translation>
Entsprechenserklärung zum Deutschen Corporate Governance Kodex
</German_Translation>
- Desired JSON Output:
{{
  "context_match": "Yes",
  "explanation": "The context is preserved. Both are functionally identical titles referring to the same document."
}}
---
**Example 3: Context Matches (Same Event)**
<Original_English>
The Board dealt with the issue.
</Original_English>
<German_Translation>
Der Vorstand befasste sich mit dem Thema.
</German_Translation>
- Desired JSON Output:
{{
  "context_match": "Yes",
  "explanation": "The context is preserved; both texts describe the same event of the Board handling the issue."
}}
---

# YOUR TASK
Analyze the following paragraphs for a fundamental change in the narrative or the roles of the subjects and provide your JSON response.

<Original_English>
{eng_text}
</Original_English>

<German_Translation>
{ger_text}
</German_Translation>
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        content = response['message']['content']
        json_str = content[content.find('{'):content.rfind('}')+1]
        return json.loads(json_str)
    except Exception as e:
        return {"context_match": "Error", "explanation": str(e)}

def evaluate_table_pair(eng_html: str, ger_html: str, model_name: str = 'mistral:latest'):
    prompt = f"""
    You are a data auditor comparing two HTML tables. Find critical differences in structure, data, or headers.
    - English Table: {eng_html}
    - German Table: {ger_html}
    Respond in JSON: {{"error_type": "None" or "Structural_Mismatch" or "Data_Mismatch" or "Header_Error", "explanation": "..."}}
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        content = response['message']['content']
        json_str = content[content.find('{'):content.rfind('}')+1]
        return json.loads(json_str)
    except Exception as e:
        return {"error_type": "System Error", "explanation": str(e)}