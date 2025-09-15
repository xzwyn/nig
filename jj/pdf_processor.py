# pdf_processor.py (updated function)

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, ListItem, Text, Table

def map_element_type(element):
    # ... (this helper function remains the same)
    if isinstance(element, Title):
        return 'heading'
    if isinstance(element, Table):
        return 'table'
    if isinstance(element, NarrativeText):
        return 'paragraph'
    if isinstance(element, ListItem):
        return 'list_item'
    if isinstance(element, Text):
        return 'paragraph' 
    return 'other'

def extract_structural_elements(pdf_path: str, language: str, strategy="hi_res"):
    """
    Extracts and classifies elements from a PDF using 'unstructured',
    now with explicit language support.
    """
    print(f"\n--- PDF Processing Stage (using 'unstructured' with language='{language}') ---")
    try:
        raw_elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            infer_table_structure=True,
            languages=[language] # Pass the specified language
        )
    except Exception as e:
        print(f"An error occurred with the 'unstructured' library: {e}")
        return []

    # --- Convert the output from 'unstructured' to our desired format ---
    # ... (the rest of the function remains the same)
    final_elements = []
    id_counter = 0
    
    for element in raw_elements:
        element_type = map_element_type(element)
        text = element.text.strip()
        
        if text and element_type != 'other':
            item = {
                'id': f'e_{id_counter}',
                'text': text,
                'type': element_type,
                'page': element.metadata.page_number or 0 
            }
            if element_type == 'table':
                item['html'] = element.metadata.text_as_html
            
            final_elements.append(item)
            id_counter += 1
            
    print(f"Processing complete. Found {len(final_elements)} classified elements.")
    return final_elements