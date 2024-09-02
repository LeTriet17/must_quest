
from langchain.docstore.document import Document


class DocumentWithVSId(Document):
    """
    Document with vector store id
    """
    id: str = None
    # score: float = 3.0
