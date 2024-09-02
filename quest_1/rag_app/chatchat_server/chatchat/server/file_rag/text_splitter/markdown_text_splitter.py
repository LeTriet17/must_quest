import re
from chatchat.server.utils import get_Embeddings
import os
from functools import lru_cache
from chatchat.configs import (
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    logger,
    log_verbose,
    TEXT_SPLITTER_NAME,
    DEFAULT_EMBEDDING_MODEL,
    TABLE_SUMMARIZE_CONCURRENT
)
import importlib
import langchain_community.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from chatchat.server.knowledge_base.kb_summary_api import summarize_text
import ast

def is_table_line(line):
    return re.match(r'\|.*?\|', line) is not None

def table_aware_markdown_splitter(docs, chunk_size=1000, chunk_overlap=200):
    """
    Split the markdown text into LangChain Documents while preserving table formatting.
    """
    table_starts = []
    table_ends = []
    markdown_text = docs.page_content
    doc_id = docs.metadata.get("id", 0)
    lines = markdown_text.splitlines()
    embeddings = get_Embeddings(embed_model=DEFAULT_EMBEDDING_MODEL)
    # Find the start and end lines of tables
    in_table = False
    for i, line in enumerate(lines):
        if is_table_line(line) and not in_table:
            in_table = True
            table_starts.append(i)
        elif is_table_line(line) and in_table:
            pass
        elif not is_table_line(line) and in_table:
            in_table = False
            table_ends.append(i - 1)
    if in_table:
        table_ends.append(len(lines) - 1)

    # Combine table start and end indices
    table_ranges = list(zip(table_starts, table_ends))

    semantic_chunker = SemanticChunker(
    embeddings, breakpoint_threshold_type="percentile"
    )
    # Split the text while preserving tables
    documents = []
    tables = []
    start = 0
    for table_start, table_end in table_ranges:
        # Split the text before the table
        if start < table_start:
            pre_table_text = '\n'.join(lines[start:table_start])
            pre_table_chunks = semantic_chunker.create_documents([pre_table_text])
            for chunk in pre_table_chunks:
                chunk.metadata["id"] = doc_id
                documents.append(chunk)

        # Add the table as a single document
        table_text = '\n'.join(lines[table_start:table_end + 1])
        tables.append(table_text)

        # Update the start index
        start = table_end + 1
    logger.info(f"Splitting tables: {tables}")
    try:
        for table_start_idx in range(0, len(tables), TABLE_SUMMARIZE_CONCURRENT):
            tables_converted_text = ','.join(tables[table_start_idx:table_start_idx + TABLE_SUMMARIZE_CONCURRENT])
            tables_text = f'[{tables_converted_text}]'
            tables_summarize = summarize_text(tables_text)
            # convert string list into list '[]' -> []
            tables_summarize = ast.literal_eval(tables_summarize)
            for table_idx, table_text in enumerate(tables_summarize): 
                table_doc = Document(page_content=table_text, metadata={"id": doc_id, "table":tables[table_start_idx + table_idx]})
                documents.append(table_doc)
    except Exception as e:
        logger.error(f"Error summarizing tables: {e}")

    # Split the remaining text after the last table
    if start < len(lines):
        post_table_text = '\n'.join(lines[start:])
        post_table_chunks = semantic_chunker.create_documents([post_table_text])
        for chunk in post_table_chunks:
            chunk.metadata["id"] = doc_id
            documents.append(chunk)
    # Filder document with page_content != ''
    documents = list(filter(lambda x: x.page_content != '', documents))
    return documents