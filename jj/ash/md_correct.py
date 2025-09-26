import os

input_filename = "extracted_content.md"
output_filename = "corrected_content.md"

def correct_paragraphs(content):
    paragraphs = content.split('\n\n')
    
    if not paragraphs:
        return ""
    corrected_paragraphs = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        current_paragraph = paragraphs[i].strip()
        previous_paragraph = corrected_paragraphs[-1].strip()

        if previous_paragraph and not previous_paragraph.startswith('#') and not previous_paragraph.endswith('.'):
            corrected_paragraphs[-1] = f"{previous_paragraph} {current_paragraph}"
        else:
            corrected_paragraphs.append(current_paragraph)
            
    return '\n\n'.join(corrected_paragraphs)

if __name__ == "__main__":
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        print("Please run your extraction script (doc.py) first to generate it.")
    else:
        print(f"Reading '{input_filename}' to correct paragraphs...")
        
        with open(input_filename, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        corrected_content = correct_paragraphs(original_content)
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(corrected_content)
            
        print(f" Corrected content has been saved to '{output_filename}'")