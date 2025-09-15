import ollama
import json
from evaluator import evaluate_translation_pair, check_context_mismatch, evaluate_table_pair

def _agent2_validate_finding(eng_text, ger_text, error_type, explanation, model_name='mistral:latest'):
    prompt = f"""
# ROLE
You are a Senior Quality Reviewer. Your job is to reject any finding that is not a fatal, objectively verifiable error.

# TASK
Critically evaluate a finding from an automated system. Reject anything that is subjective or a minor issue.

# INSTRUCTIONS
1.  **Confirm** a finding ONLY if it meets one of these strict criteria:
    - It is a clear factual mistranslation of a **number or name**.
    - It is a mistranslation to a word's **direct opposite** or alternative meaning.
    - It is a **structural omission**, meaning an item was dropped from a list that was explicitly enumerated (e.g., "three items," "four steps").

2.  **Reject** all other findings. Especially reject omissions that are just condensed phrasing or paraphrasing.If the numbers are same, REJECT it. If the error is about "potential confusion" but not a structural break, REJECT it.

3.  Provide a clear reason for your verdict in a strict JSON format.

# EXAMPLES
---
**Example 1: Correct Confirmation (Structural Omission)**
- Initial Finding: An omission was flagged because the second of 'three key areas' was missing.
- Your Analysis: The finding points to an enumerated list being broken. This is a critical structural error.
- Desired JSON Output:
{{
  "verdict": "Confirm",
  "reasoning": "The finding correctly identifies a structural omission where an enumerated list was broken. This is a critical error."
}}
---
**Example 2: Correct Rejection (Not a Structural Omission)**
- Initial Finding: An omission was flagged because 'the committee discussed the details' was translated as 'der Ausschuss besprach die Punkte.'
- Your Analysis: The phrasing is condensed, but no enumerated list was broken. This is not a critical error.
- Desired JSON Output:
{{
  "verdict": "Reject",
  "reasoning": "The finding is a false positive. The translation is a valid paraphrase and does not break any explicit list structure."
}}
---

# YOUR TASK
Review the finding against the strict criteria and provide your JSON verdict.

- Original English: "{eng_text}"
- German Translation: "{ger_text}"
- Flagged Error Type: "{error_type}"
- System's Explanation: "{explanation}"
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0}
        )
        content = response['message']['content']
        json_str = content[content.find('{'):content.rfind('}')+1]
        if not json_str:
            raise ValueError("No JSON object found in the raw response.")
        
        verdict_json = json.loads(json_str)
        is_confirmed = verdict_json.get("verdict", "Reject").lower() == "confirm"
        reasoning = verdict_json.get("reasoning", "No reasoning found.")
        return is_confirmed, reasoning, content.strip()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  - Agent 2 failed: {e}")
        return False, f"System error: {e}", "{}"
    except Exception as e:
        print(f"  - Agent 2 encountered a non-parsing error: {e}")
        return False, "System error (non-parsing issue)", "{}"

def run_evaluation_pipeline(aligned_pairs: list, model_name: str = 'mistral:latest'):
    print("\n--- Agentic Evaluation Stage ---")

    for idx, (eng_elem, ger_elem) in enumerate(aligned_pairs):
        print(f"\n{'='*60}")
        eng_id = eng_elem['id'] if eng_elem else 'N/A'
        ger_id = ger_elem['id'] if ger_elem else 'N/A'
        print(f"Processing pair {idx}: ENG [{eng_id}] | GER [{ger_id}]")
        
        if eng_elem and not ger_elem:
            yield { "type": f"Omitted {eng_elem.get('type', 'Item')}", "english_text": eng_elem['text'], "german_text": "---", "suggestion": "Missing from German document", "page": eng_elem['page'] }
            continue
        if not eng_elem and ger_elem:
            yield { "type": f"Added {ger_elem.get('type', 'Item')}", "english_text": "---", "german_text": ger_elem['text'], "suggestion": "Extra item in German document", "page": ger_elem['page'] }
            continue

        if eng_elem and ger_elem:
            if eng_elem.get('type') == 'table':
                finding = evaluate_table_pair(eng_elem.get('html', ''), ger_elem.get('html', ''), model_name)
                print(f"  - Agent 1 Result (Table): {finding.get('error_type', 'None')}")
                if finding.get("error_type", "None") not in ["None", "System Error"]:
                    yield { "type": f"Table Error: {finding.get('error_type')}", "english_text": f"Table on page {eng_elem['page']}", "german_text": f"Table on page {ger_elem['page']}", "suggestion": finding.get("explanation"), "page": eng_elem['page'] }
                continue

            eng_text = eng_elem['text']
            ger_text = ger_elem['text']
            print(f"Evaluating: \"{eng_text[:60].strip()}...\"")

            finding = evaluate_translation_pair(eng_text, ger_text, model_name)
            error_type = finding.get("error_type", "None")
            print(f"Agent 1 Result: {error_type}")
            if error_type not in ["None", "System Error"]:
                print(f"Agent 1 JSON Response: {json.dumps(finding, indent=2)}")
                
                is_confirmed, reasoning, agent2_raw_response = _agent2_validate_finding(eng_text, ger_text, error_type, finding.get("explanation"), model_name)
                verdict_str = 'Accept' if is_confirmed else 'Reject'
                print(f"Agent 2 Response: {verdict_str}")
                print(f"Agent 2 JSON Response: {agent2_raw_response}")
                
                if is_confirmed:
                    yield { "type": error_type, "english_text": eng_text, "german_text": ger_text, "suggestion": finding.get("suggestion"), "page": eng_elem['page'], "original_phrase": finding.get("original_phrase"), "translated_phrase": finding.get("translated_phrase") }
            else:
                context_result = check_context_mismatch(eng_text, ger_text, model_name)
                context_match_verdict = context_result.get('context_match', 'Error')
                print(f"Agent 3 Context Match: {context_match_verdict}")
                print(f"Agent 3 JSON Response: {json.dumps(context_result, indent=2)}")
                if context_match_verdict.lower() == "no":
                    yield { "type": "Context Mismatch", "english_text": eng_text, "german_text": ger_text, "suggestion": context_result.get("explanation"), "page": eng_elem['page'] }

    print(f"\n--- Evaluation complete. ---")