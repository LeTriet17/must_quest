import os

from .basic_config import DATA_PATH

# Number of matched vectors in the knowledge base
VECTOR_SEARCH_TOP_K = 10

# Relevance threshold for knowledge base matching, range is between 0-1. The smaller the SCORE, the higher the relevance. Setting it to 1 means no filtering. It is recommended to set it around 0.5.
VECTOR_SCORE_THRESHOLD = 0.1

KEYWORD_SCORE_THRESHOLD = 0.6
# Default search engine. Options: bing, duckduckgo, metaphor
DEFAULT_SEARCH_ENGINE = "duckduckgo"

# Number of matched results for search engines
SEARCH_ENGINE_TOP_K = 10

# Introduction for each knowledge base, used for displaying and Agent calling during initialization. If not written, there will be no introduction, and it will not be called by the Agent.
KB_INFO = {
    "samples": "Answers to issues about this project",
}

ROOT_KB_NAME = "knowledge_base"

# Default storage path for knowledge bases
KB_ROOT_PATH = os.path.join(DATA_PATH, ROOT_KB_NAME)
if not os.path.exists(KB_ROOT_PATH):
    os.makedirs(KB_ROOT_PATH, exist_ok=True)

# Optional vector store types and corresponding configurations
kbs_config = {
    "mongodb": {
        "db_name": "odd",
        "knowledge_bases": "knowledge_bases",
        "knowledge_files": "knowledge_files",
        "messages": "messages",
        "embeddings": "embeddings",
        "topics": "topics",
        "users": "users",
        "questionaires": "questionaires",
    }
}

ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"  

VECTOR_INDEX_KEY="document"

ATLAS_VECTOR_TEXT_INDEX_NAME = "text"

ATLAS_TEXT_INDEX_KEY="document"

# TextSplitter configuration items, if you do not understand their meaning, do not modify them.
text_splitter_dict = {
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on": [
            ("#", "head1"),
            ("##", "head2"),
            ("###", "head3"),
            ("####", "head4"),
        ]
    },
}

# TEXT_SPLITTER name
TEXT_SPLITTER_NAME = "MarkdownHeaderTextSplitter"

# Page separator for LLAMA parsing
PAGE_SEPARATOR = "\nPAGE_SEPARATOR\n"

# Breakpoint threshold type for semantic chunking
BREAKPOINT_THRESHOLD_TYPE = "percentile"

# Breakpoint threshold amount for semantic chunking
BREAKPOINT_THRESHOLD_AMOUNT = 0.90