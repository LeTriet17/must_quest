import os


# Default LLM name to use
DEFAULT_LLM_MODEL = "gpt-4o"

MINI_GPT_MODEL = "gpt-4o-mini"

MODEL_NAME_1 = "Agent A"

MODEL_NAME_2 = "Agent B"

# Anthropic model name
DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20240229"

# Default embedding model name to use
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

# Voyage Embedding model name
DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-finance-2"

EMBEDDING_DIM = 1024

# Number of historical dialogue turns
HISTORY_LEN = 3

# Maximum length supported by large models. If not filled in, use the default maximum length of the model. If filled in, it is the maximum length set by the user.
MAX_TOKENS = 2048


# TOP_P = 0.95 # ChatOpenAI does not support this parameter currently

TABLE_SUMMARIZE_CONCURRENT = 30

SUPPORT_AGENT_MODELS = [
    "openai-api",
]


ENTITY_RANGE = 3

ITERATIONS = 3

CONTENT_CATEGORY = "Financial Report"


LLM_MODEL_CONFIG = {
    "llm_model": {
        DEFAULT_LLM_MODEL: {
            "temperature": 0.1,
            "max_tokens": 4096,
            "history_len": 10,
            "prompt_name": "rag_query",
            "callbacks": True
        },
        DEFAULT_ANTHROPIC_MODEL: {
            "temperature": 0.1,
            "max_tokens": 4096,
            "history_len": 10,
            "prompt_name": "rag_query",
            "callbacks": True
        },
    }
}

OPENAI_BASE_URL = "https://oai.helicone.ai/v1"

ANSWER_CONFIDENT_THRESHOLD = 0.7

DENY_CONFIDENT_THRESHOLD = 0.5

ANSWER_RETRY_TIMES = 3