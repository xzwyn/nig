import json
from evaluator import evaluate_translation_pair, check_context_mismatch, evaluate_table_pair
from azure_client import chat  # NEW: use Azure OpenAI

# def _agent2_validate_finding(
#     eng_text: str,
#     ger_text: str,
#     error_type: str,
#     explanation: str,
#     model_name: str | None = None,
# ):
def _agent2_validate_finding(
    eng_text: str,
    ger_text: str,
    error_type: str,
    explanation: str,
    model_name: str | None = None,
):
    """
    Second-stage reviewer.  Confirms only truly fatal errors and rejects
    false positives.
    """
    prompt = f"""
## ROLE
**Senior Quality Reviewer** – you are the final gatekeeper of EN→DE
translation findings.

## TASK
Decide whether the finding delivered by Agent-1 must be *Confirmed* or
*Rejected*.

## INSTRUCTIONS
1. Eligible error_type values are **exactly**:
   • "Mistranslation"  
   • "Omission"

2. Confirm only when the evidence is unmistakable:
   • Mistranslation
       – number mismatch (digit or word)  
       – polarity flip / opposite meaning  
       – actor/role inversion  
   • Omission
       – English states an explicit count (“two”, “three”, “both” …) **or**
         lists concrete items, and at least one item is *truly* missing in
         German (not conveyed by paraphrase).

3. Reject when:
   • Difference is stylistic or synonymous.  
   • Proper names / document titles are rendered with an accepted German
     equivalent (e.g. “Nichtfinanzielle Erklärung”).  
   • Alleged omission is actually present via paraphrase.  

## OUTPUT ‑ JSON ONLY
json {{ "verdict" : "Confirm" | "Reject", "reasoning": "" }}

## MATERIAL TO REVIEW
English text:
\"\"\"{eng_text}\"\"\"

German text:
\"\"\"{ger_text}\"\"\"

Agent-1 proposed:
  error_type : {error_type}
  explanation: {explanation}

## YOUR RESPONSE
Return the JSON object only – no extra text.
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        )
        j0, j1 = content.find("{"), content.rfind("}") + 1
        verdict_json = json.loads(content[j0:j1])
        is_confirmed = verdict_json.get("verdict", "").lower() == "confirm"
        reasoning = verdict_json.get("reasoning", "")
        return is_confirmed, reasoning, content.strip()
    except (ValueError, json.JSONDecodeError) as exc:
        print("  - Agent-2 JSON parse error:", exc)
        return False, f"System error: {exc}", "{}"
    except Exception as exc:
        print("  - Agent-2 unexpected error:", exc)
        return False, "System error (non-parsing issue)", "{}"


def run_evaluation_pipeline(aligned_pairs: list, model_name: str = None):
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