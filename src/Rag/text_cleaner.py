import re
import string
from typing import List
from langchain_core.documents import Document


def clean_text(chunks: List[Document]) -> List[Document]:
    cleaned_documents = []

    for doc in chunks:
        text = doc.page_content
        lines = text.splitlines()

        cleaned_lines = []
        for line in lines:
            # Strip leading/trailing whitespace from each line
            line = line.strip()
            # Replace multiple spaces or tabs with a single space
            line = re.sub(r"[ \t]+", " ", line)
            # Remove non-printable characters
            line = "".join(char for char in line if char in string.printable)
            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)

        # Create a new Document with cleaned text and original metadata
        cleaned_doc = Document(page_content=cleaned_text, metadata=doc.metadata)
        cleaned_documents.append(cleaned_doc)

    return cleaned_documents