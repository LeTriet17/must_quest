import os


# Default LLM name to use
DEFAULT_LLM_MODEL = "gpt-4o"

# Default embedding model name to use
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

# AgentLM model name (optional, if specified, it locks the model for the Chain after entering the Agent, if not specified, it defaults to LLM_MODELS[0])
Agent_MODEL = None

# Number of historical dialogue turns
HISTORY_LEN = 3

# Maximum length supported by large models. If not filled in, use the default maximum length of the model. If filled in, it is the maximum length set by the user.
MAX_TOKENS = 2048

# Universal dialogue parameters for LLM
TEMPERATURE = 0.2

# TOP_P = 0.95 # ChatOpenAI does not support this parameter currently

TABLE_SUMMARIZE_CONCURRENT = 30

SUPPORT_AGENT_MODELS = [
    "openai-api",
]


ENTITY_RANGE = 3

ITERATIONS = 3

CONTENT_CATEGORY = "Financial Report"


LLM_MODEL_CONFIG = {
    # Intent recognition does not need output, the model backend knows it
    "preprocess_model": {
        DEFAULT_LLM_MODEL: {
            "temperature": 0.05,
            "max_tokens": 4096,
            "history_len": 100,
            "prompt_name": "default",
            "callbacks": False
        },
    },
    "llm_model": {
        DEFAULT_LLM_MODEL: {
            "temperature": 0.2,
            "max_tokens": 4096,
            "history_len": 10,
            "prompt_name": "default",
            "callbacks": True
        },
    },
    "action_model": {
        DEFAULT_LLM_MODEL: {
            "temperature": 0.01,
            "max_tokens": 4096,
            "prompt_name": "ChatGLM3",
            "callbacks": True
        },
    },
    "postprocess_model": {
        DEFAULT_LLM_MODEL: {
            "temperature": 0.01,
            "max_tokens": 4096,
            "prompt_name": "default",
            "callbacks": True
        }
    },
    "image_model": {
        "sd-turbo": {
            "size": "256*256",
        }
    },
    "multimodal_model": {
        "qwen-vl": {}
    },
}

# You can start the model service through loom/xinference/oneapi/fastchat, and then configure its URL and KEY.
#   - platform_name can be filled arbitrarily, just don't repeat it
#   - platform_type optional: openai, xinference, oneapi, fastchat. Some functional distinctions may be made in the future based on the platform type
#   - Fill in the models deployed by the framework to the corresponding list. Different frameworks can load models with the same name, and the project will automatically do load balancing.

MODEL_PLATFORMS = [
    {
        "platform_name": "openai-api",
        "platform_type": "openai-api",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "API-KEY-HERE",
        "api_concurrencies": 10,
        "llm_models": [
            "gpt-4o",
        ],
        "embed_models": [
            # "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
        "image_models": [],
        "reranking_models": [],
        "speech2text_models": [],
        "tts_models": [],
    },


]

LOOM_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loom.yaml")

# Tool configuration
TOOL_CONFIG = {
    "search_local_knowledgebase": {
        "use": False,
        "top_k": 5,
        "score_threshold": 0.5,
        "conclude_prompt": {
            "with_result":
                '<Instruction>Based on the known information, answer the question concisely and professionally. If you cannot find the answer from it, say "Unable to answer the question based on known information", and do not allow fabricated components in the answer. Please use Chinese in the answer.</Instruction>\n'
                '<Known Information>{{ context }}</Known Information>\n'
                '<Question>{{ question }}</Question>\n',
            "without_result":
                'Please answer my question based on my query:\n'
                '{{ question }}\n'
                'Please note that you must emphasize after the answer that your response is based on your experience rather than reference materials.\n',
        }
    },
    "search_internet": {
        "use": False,
        "search_engine_name": "bing",
        "search_engine_config":
            {
                "bing": {
                    "result_len": 3,
                    "bing_search_url": "https://api.bing.microsoft.com/v7.0/search",
                    "bing_key": "",
                },
                "metaphor": {
                    "result_len": 3,
                    "metaphor_api_key": "",
                    "split_result": False,
                    "chunk_size": 500,
                    "chunk_overlap": 0,
                },
                "duckduckgo": {
                    "result_len": 3
                }
            },
        "top_k": 10,
        "verbose": "Origin",
        "conclude_prompt":
            "<Instruction>This is the information found on the internet. Please extract and organize the information to answer the question concisely. If you cannot find the answer from it, say 'Unable to find content that can answer the question'. "
            "</Instruction>\n<Known Information>{{ context }}</Known Information>\n"
            "<Question>\n"
            "{{ question }}\n"
            "</Question>\n"
    },
    "arxiv": {
        "use": False,
    },
    "shell": {
        "use": False,
    },
    "weather_check": {
        "use": False,
        "api_key": "S8vrB4U_-c5mvAMiK",
    },
    "search_youtube": {
        "use": False,
    },
    "wolfram": {
        "use": False,
        "appid": "",
    },
    "calculate": {
        "use": False,
    },
    "vqa_processor": {
        "use": False,
        "model_path": "your model path",
        "tokenizer_path": "your tokenizer path",
        "device": "cuda:1"
    },
    "text2images": {
        "use": False,
    },

}