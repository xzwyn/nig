from pathlib import Path
from typing import List, Dict, Any

ContentItem = Dict[str, Any]

def save_to_markdown(content: List[ContentItem], filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in content:
            if item['type'] in {'title', 'sectionHeading', 'subheading'}:
                f.write(f"## {item['text']}\n\n")
            else:
                f.write(f"{item['text']}\n\n")