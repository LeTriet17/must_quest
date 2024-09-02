from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func
from chatchat.server.db.base import Base

class SummaryChunkModel(Base):
    """
    Chunk summary model, used to store chunks of each doc_id in file_doc,
    Data Source:
        User Input: User uploads files, can fill in file descriptions, generate doc_id in file_doc, and store it in summary_chunk
        Program Auto-Split: The page information stored in the meta_data field of the file_doc table is split according to the page number of each page, and a custom prompt generates the summary text, and the corresponding doc_id associated with the page number is stored in summary_chunk
    Subsequent Tasks:
        Vector Store Construction: Create an index for summary_context in the database table summary_chunk, construct a vector store, and meta_data is the metadata of the vector store (doc_ids)
        Semantic Association: Calculate semantic similarity through user input description, automatically split summary text, and compute semantic similarity
    """
    __tablename__ = 'summary_chunk'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    kb_name = Column(String(50), comment='Knowledge Base Name')
    summary_context = Column(String(255), comment='Summary Text')
    summary_id = Column(String(255), comment='Summary Vector ID')
    doc_ids = Column(String(1024), comment="Vector Store ID Association List")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        return (f"<SummaryChunk(id='{self.id}', kb_name='{self.kb_name}', summary_context='{self.summary_context}',"
                f" doc_ids='{self.doc_ids}', metadata='{self.metadata}')>")
