import os

from .basic_config import DATA_PATH

# Default vector store/full-text search engine type. Options: faiss, milvus (offline) & zilliz (online), pgvector, full-text search engine es
DEFAULT_VS_TYPE = "faiss"

# Number of cached vector stores (for FAISS)
CACHED_VS_NUM = 1

# Number of cached temporary vector stores (for FAISS), used for file dialogs
CACHED_MEMO_VS_NUM = 10

# Single text segment length in the knowledge base (not applicable to MarkdownHeaderTextSplitter)
CHUNK_SIZE = 2000

# Overlapping length of adjacent texts in the knowledge base (not applicable to MarkdownHeaderTextSplitter)
OVERLAP_SIZE = 400

# Number of matched vectors in the knowledge base
VECTOR_SEARCH_TOP_K =  10

# Relevance threshold for knowledge base matching, range is between 0-1. The smaller the SCORE, the higher the relevance. Setting it to 1 means no filtering. It is recommended to set it around 0.5.
SCORE_THRESHOLD = 0.5

# Default search engine. Options: bing, duckduckgo, metaphor
DEFAULT_SEARCH_ENGINE = "duckduckgo"

# Number of matched results for search engines
SEARCH_ENGINE_TOP_K = 10

# Whether to enable Chinese title enhancement and related configuration
# By adding title judgment, determine which texts are titles and mark them in metadata.
# Then merge the text with the title of the previous level to enhance the text information.
ZH_TITLE_ENHANCE = False

# PDF OCR control: Only perform OCR on images that exceed a certain proportion of the page (image width/page width, image height/page height).
# This can avoid interference from small images in the PDF and improve the processing speed of non-scanned PDFs.
PDF_OCR_THRESHOLD = (0.6, 0.6)

# Introduction for each knowledge base, used for displaying and Agent calling during initialization. If not written, there will be no introduction, and it will not be called by the Agent.
KB_INFO = {
    "samples": "Answers to issues about this project",
}

# Usually, the following content does not need to be changed

# Default storage path for knowledge bases
KB_ROOT_PATH = os.path.join(DATA_PATH, "knowledge_base")
if not os.path.exists(KB_ROOT_PATH):
    os.mkdir(KB_ROOT_PATH)

# Default storage path for the database.
# If using SQLite, you can directly modify DB_ROOT_PATH; if using other databases, please directly modify SQLALCHEMY_DATABASE_URI.
DB_ROOT_PATH = os.path.join(KB_ROOT_PATH, "info.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_ROOT_PATH}"

# Optional vector store types and corresponding configurations
kbs_config = {
    "faiss": {},
    # "milvus": {
    #     "host": "127.0.0.1",
    #     "port": "19530",
    #     "user": "",
    #     "password": "",
    #     "secure": False,
    # },
    # "pg": {
    #     "connection_uri": "postgresql://postgres:postgres@127.0.0.1:5432/langchain_chatchat",
    # },
    # "milvus_kwargs": {
    #     "search_params": {"metric_type": "L2"},  # Add search_params here
    #     "index_params": {"metric_type": "L2", "index_type": "HNSW"}  # Add index_params here
    # },
    # "chromadb": {
    #     "host": "127.0.0.1",
    #     "port": "8000",
# }
}
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

# Custom vocabulary file for the embedding model
EMBEDDING_KEYWORD_FILE = "embedding_keywords.txt"

# LLAMA CLOUD API key
LLAMA_API_KEY = "llx-dTmPzzwXMjuUP0Jh4I1C6dpFiRAN7EnO06JpSlA37Nc88ENH"