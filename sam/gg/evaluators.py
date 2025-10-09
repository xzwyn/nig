# src/evaluation/evaluators.py
import json
from src.clients.azure_client import chat  

def evaluate_translation_pair(eng_text: str, ger_text: str, model_name=None):
    prompt = f"""
## ROLE
You are the Primary Translation Auditor for EN→DE corporate reports.

## TASK
Identify only the two fatal error categories below and output ONE JSON object.

## ERROR TYPES YOU MAY REPORT
1. Mistranslation
   • Wrong numeric value (digits, words, units, decimals, percentages)
   • Polarity flip / negation error (e.g., required ↔ not required)
   • Change of actor or agency (who did/decided/informed whom)

2. Omission
   The English text states a concrete count (“two”, “three”, “both”, “either”) or lists specific items, and at least one required element is missing in German.

Do not flag: stylistic differences, safe synonyms, acceptable German report titles (“Nichtfinanzielle Erklärung”, “Erklärung zur Unternehmensführung” etc.), benign reordering, or tense/voice changes that preserve actor and meaning.

If no fatal error is found, return error_type "None".

If multiple fatal errors exist, choose the most impactful; if tied, prefer "Mistranslation".

## JSON OUTPUT SCHEMA
json {{ "error_type" : "Mistranslation" | "Omission" | "None", "original_phrase" : "", "translated_phrase": "", "explanation" : "<≤40 words>", "suggestion" : "" }}

## POSITIVE EXAMPLES
1 · Mistranslation (number)
EN “Revenue increased by 2.3 million.”
DE “Der Umsatz stieg um 2,8 Millionen.”
→ error_type “Mistranslation”, original “2.3 million”, translated “2,8 Millionen”

2 · Mistranslation (polarity)
EN “The audit is not required.”
DE “Die Prüfung ist erforderlich.”
→ error_type “Mistranslation”, original “not required”, translated “erforderlich”

3 · Mistranslation (actor/agency)
EN “The company was notified by the regulator.”
DE “Das Unternehmen informierte die Aufsichtsbehörde.”
→ error_type “Mistranslation”, original “was notified by the regulator”, translated “informierte die Aufsichtsbehörde”

4 · Omission (enumeration/count)
EN “Both measures will apply: cost cap and hiring freeze.”
DE “Es gilt die Einstellungsstop.”
→ error_type “Omission”, original “cost cap”, translated “”

5 · None (acceptable variation)
EN “The report is comprehensive.”
DE “Der Bericht ist umfassend.”
→ error_type “None”

## TEXTS TO AUDIT
<Original English>
{eng_text}
</Original English>

<German Translation>
{ger_text}
</German Translation>

## YOUR RESPONSE
Return the JSON object only—no extra text, no markdown.

## NOTES
- Compare all numbers, signs, and units (%, bps, million/Mio., billion/Mrd.).
- Treat passive/active voice as fine unless the responsible actor changes.
- For omissions, ensure every counted or listed element appears in German.
- Keep “explanation” concise; “suggestion” should minimally correct the German (or note the missing item).
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
        print(f"evaluate_translation_pair → {exc}")
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
