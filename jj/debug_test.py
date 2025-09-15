# debug_test.py - Test script to debug your evaluation issues

from evaluator import (
    evaluate_translation_pair, 
    evaluate_translation_pair_alternative,
    evaluate_with_explicit_examples,
    test_model_response
)
from agent import debug_single_pair

def test_with_obvious_errors():
    """Test with pairs that definitely have errors to see if the model can detect them."""
    
    test_cases = [
        {
            "name": "Obvious Mistranslation",
            "english": "The company reported profits of 5 million dollars last year.",
            "german": "Das Unternehmen meldete Verluste von 5 Millionen Dollar im letzten Jahr."  # profits -> losses
        },
        {
            "name": "Missing Information", 
            "english": "The meeting will be held on Monday at 3 PM in Conference Room A.",
            "german": "Das Meeting findet am Montag statt."  # missing time and room
        },
        {
            "name": "Wrong Numbers",
            "english": "Sales increased by 15% to reach 100 million euros.",
            "german": "Der Umsatz stieg um 25% auf 50 Millionen Euro."  # wrong percentage and amount
        },
        {
            "name": "Good Translation (should return None)",
            "english": "The Board of Directors approved the annual budget.",
            "german": "Der Vorstand genehmigte das Jahresbudget."
        }
    ]
    
    print("🧪 Testing with obvious errors...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"English: {test_case['english']}")
        print(f"German:  {test_case['german']}")
        print("-" * 60)
        
        # Test all strategies
        strategies = [
            ("Primary", evaluate_translation_pair),
            ("Alternative", evaluate_translation_pair_alternative), 
            ("Examples", evaluate_with_explicit_examples)
        ]
        
        for strategy_name, strategy_func in strategies:
            print(f"\n🔍 {strategy_name} Strategy:")
            try:
                result = strategy_func(test_case["english"], test_case["german"])
                error_type = result.get("error_type", "Unknown")
                
                if error_type == "None":
                    print("  ✅ No error detected")
                elif error_type == "System Error":
                    print(f"  ❌ System Error: {result.get('explanation', 'Unknown')}")
                else:
                    print(f"  🚨 ERROR DETECTED: {error_type}")
                    print(f"     Original: {result.get('original_phrase', 'N/A')}")
                    print(f"     Translation: {result.get('translated_phrase', 'N/A')}")
                    print(f"     Explanation: {result.get('explanation', 'N/A')}")
                    
            except Exception as e:
                print(f"  💥 Exception: {e}")


def test_model_basic_functionality():
    """Test if the Ollama model is working at all."""
    print("🔧 Testing basic model functionality...")
    
    if not test_model_response():
        print("❌ Basic model test failed!")
        return False
        
    # Test with simple JSON response
    import ollama
    try:
        response = ollama.chat(
            model='aya',
            messages=[{
                'role': 'user', 
                'content': 'Respond with JSON: {"test": "success", "message": "model is working"}'
            }],
            options={'temperature': 0.0}
        )
        
        print(f"Raw response: {response['message']['content']}")
        
        import json
        # Try to parse JSON from response
        content = response['message']['content']
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            parsed = json.loads(json_str)
            print(f"✅ JSON parsing successful: {parsed}")
            return True
        else:
            print("❌ No valid JSON found in response")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def run_diagnostics():
    """Run complete diagnostics."""
    print("🏥 Running Translation Evaluator Diagnostics...")
    print("="*60)
    
    # Step 1: Test basic model functionality
    print("\n1️⃣ Testing Model Connectivity...")
    if not test_model_basic_functionality():
        print("🚨 Model connectivity issues detected!")
        print("   - Check if Ollama is running: ollama serve")
        print("   - Check if 'aya' model is available: ollama list")
        print("   - Try pulling the model: ollama pull aya")
        return
    
    # Step 2: Test with obvious errors
    print("\n2️⃣ Testing Error Detection...")
    test_with_obvious_errors()
    
    print("\n📋 Diagnostic Summary:")
    print("- If no errors were detected in the obvious cases above,")
    print("  the issue is likely with prompt engineering or model capability")
    print("- Try using a different model (e.g., 'llama3.1', 'mistral')")
    print("- Consider adjusting temperature and other parameters")
    

if __name__ == "__main__":
    run_diagnostics()