import json
import csv

# --- JSON Processing Configuration ---
# Roles to be ignored during extraction
IGNORED_ROLES = {"pageHeader", "pageFooter", "pageNumber"}
# Roles that are considered as headings
STRUCTURAL_ROLES = {'title', 'sectionHeading', 'subheading'}

def extract_headings_from_json(file_path):
    """
    Extracts headings, their page numbers, and their types (roles) from an Azure 
    Document Intelligence JSON file. Headings are identified based on 
    STRUCTURAL_ROLES, and IGNORED_ROLES are skipped.
    """
    headings = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paragraphs = data.get('analyzeResult', {}).get('paragraphs', [])
        
        for paragraph in paragraphs:
            role = paragraph.get('role')
            
            # Skip paragraphs with ignored roles
            if role in IGNORED_ROLES:
                continue
            
            # Process paragraphs with structural roles (headings)
            if role in STRUCTURAL_ROLES:
                content = paragraph.get('content')
                page_number = None
                bounding_regions = paragraph.get('boundingRegions')
                if bounding_regions and len(bounding_regions) > 0:
                    page_number = bounding_regions[0].get('pageNumber')
                
                headings.append({'text': content, 'page': page_number, 'type': role})
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None
        
    return headings

def main():
    """
    Main function to extract headings from English and German JSON files,
    map them (including unmatched ones), and write the data to a CSV file.
    """
    english_headings = extract_headings_from_json('english.json')
    german_headings = extract_headings_from_json('german.json')

    # Exit if any file failed to load
    if english_headings is None or german_headings is None:
        return

    # Check if the number of headings matches
    if len(english_headings) != len(german_headings):
        print("Warning: The number of headings in the two files is different.")
        print(f"English headings found: {len(english_headings)}")
        print(f"German headings found: {len(german_headings)}")
        print("Unmatched headings will be marked in the output file.")

    # Prepare data for CSV export, including unmatched headings
    mapped_headings = []
    max_len = max(len(english_headings), len(german_headings))
    
    for i in range(max_len):
        # Get English heading details or mark as unmatched
        if i < len(english_headings):
            eng_text = english_headings[i]['text']
            eng_page = english_headings[i]['page']
            eng_type = english_headings[i]['type']
        else:
            eng_text, eng_page, eng_type = 'unmatched', 'unmatched', 'unmatched'

        # Get German heading details or mark as unmatched
        if i < len(german_headings):
            deu_text = german_headings[i]['text']
            deu_page = german_headings[i]['page']
            deu_type = german_headings[i]['type']
        else:
            deu_text, deu_page, deu_type = 'unmatched', 'unmatched', 'unmatched'
            
        # *** FIX IS HERE: Dictionary keys now use spaces to match csv_columns ***
        mapped_headings.append({
            'english heading': eng_text,
            'german heading': deu_text,
            'eng page': eng_page,
            'deu page': deu_page,
            'eng type': eng_type,
            'deu type': deu_type
        })

    # Write the mapped headings to a CSV file
    csv_file = 'headings_mapping.csv'
    csv_columns = ['english heading', 'german heading', 'eng page', 'deu page', 'eng type', 'deu type']
    
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(mapped_headings)
            
        print(f"Successfully created '{csv_file}' with {len(mapped_headings)} mapped headings.")
    except Exception as e:
        print(f"An unexpected error occurred while writing the CSV file: {e}")

if __name__ == '__main__':
    main()