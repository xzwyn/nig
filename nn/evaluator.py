import json
from azure_client import chat

def evaluate_translation_pair(eng_text: str, ger_text: str, model_name=None):
    prompt = f"""
## ROLE
You are the **Primary Translation Auditor** for EN→DE corporate reports.

## TASK
Detect exactly two kinds of *fatal* errors and output ONE JSON object.

## ERROR TYPES YOU MAY REPORT
1. **Mistranslation**  
   • Wrong numeric value (digit or number word)  
   • Polarity flip / opposite meaning (e.g. *comprehensive → unvollständig*)  
   • Change of actor or role (passive “was informed” → active “initiated”)

2. **Omission**  
   The English text announces a specific count (“two”, “three”, “both” …)  
   or lists concrete items, and ≥1 of those items is missing in German.

Everything else—stylistic differences, synonyms, accepted German titles
(“Nichtfinanzielle Erklärung”, “Erklärung zur Unternehmensführung” …)—
is **not** an error.

## JSON OUTPUT SCHEMA
json {{ "error_type" : "Mistranslation" | "Omission" | "None", "original_phrase" : "", "translated_phrase": "", "explanation" : "<≤40 words>", "suggestion" : "" }}

## POSITIVE EXAMPLES
### 1 · Mistranslation (number)
EN “…held **five** meetings.”  
DE “…hielt **acht** Sitzungen.”  
→ error_type “Mistranslation”, original “five”, translated “acht”

### 2 · Omission
EN “Three focus areas: A, B, C.”  
DE “Drei Schwerpunkte: A und C.”  
→ error_type “Omission”, original “B”, translated ""

## NEGATIVE EXAMPLE (should be *None*)
EN “Non-Financial Statement”  
DE “Nichtfinanzielle Erklärung”

## TEXTS TO AUDIT
<Original English>
{eng_text}
</Original English>

<German Translation>
{ger_text}
</German Translation>

## YOUR RESPONSE
Return the JSON object only—no extra text, no markdown.
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        if j0 != -1 and j1 != -1:
            return json.loads(content[j0:j1])
        return {"error_type": "System Error",
                "explanation": "No JSON object in LLM reply."}
    except Exception as exc:
        print("evaluate_translation_pair →", exc)
        return {"error_type": "System Error", "explanation": str(exc)}


def check_context_mismatch(eng_text: str, ger_text: str, model_name: str = None):
    prompt = f"""
ROLE: Narrative-Integrity Analyst

Goal: Decide if the German text tells a **different story** from the
English.  “Different” means a change in
• WHO does WHAT to WHOM
• factual outcome or direction of action
• polarity (e.g. “comprehensive” ↔ “unvollständig”)

Ignore style, word order, or minor re-phrasing.

Respond with JSON:

{{
  "context_match": "Yes" | "No",
  "explanation":  "<one concise sentence>"
}}

Examples
--------
1) Role reversal (should be No)
EN  Further, the committee *was informed* by the Board …
DE  Darüber hinaus *leitete der Ausschuss eine Untersuchung ein* …
→ roles flipped ⇒ "No"

2) Identical meaning (Yes)
EN  Declaration of Conformity with the German Corporate Governance Code
DE  Entsprechenserklärung zum Deutschen Corporate Governance Kodex
→ "Yes"

Analyse the following text pair and respond with the JSON only.

<Original_English>
{eng_text}
</Original_English>

<German_Translation>
{ger_text}
</German_Translation>
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        return json.loads(content[j0:j1])
    except Exception as exc:
        return {"context_match": "Error", "explanation": str(exc)}


def evaluate_table_pair(eng_html: str, ger_html: str, model_name: str = None):
    prompt = f"""
You are a data auditor comparing two HTML tables.  Report ONLY critical
differences in JSON:

{{"error_type": "None" | "Structural_Mismatch" | "Data_Mismatch" |
  "Header_Error", "explanation": "..." }}

- English Table: {eng_html}
- German  Table: {ger_html}
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        ).strip()

        j0, j1 = content.find("{"), content.rfind("}") + 1
        return json.loads(content[j0:j1])
    except Exception as exc:
        return {"error_type": "System Error", "explanation": str(exc)}
