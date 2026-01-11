import re

def clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
    return text