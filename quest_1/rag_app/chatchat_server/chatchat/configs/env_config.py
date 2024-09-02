import os
from dotenv import load_dotenv
from chatchat.configs import logger
load_dotenv()
APP_ENV = os.environ.get("APP_ENV", None)
logger.info(f"Current APP_ENV: {APP_ENV}")
MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING", None)
SIGNIN_ENDPOINT = os.environ.get("SIGNIN_ENDPOINT", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY", None)
AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME", None)
AWS_REGION = os.environ.get("AWS_REGION", None)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", None)

DEPLOYMENT_ENV = os.environ.get("DEPLOYMENT_ENV", None)

JWT_PUBLIC_KEY = os.environ.get("JWT_PUBLIC_KEY", None)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", None)

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))

OVERLAP_SIZE = int(os.environ.get("OVERLAP_SIZE", 100))

MIN_WORDS = int(os.environ.get("MINWORD", 10))

TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.2))

HELICONE_API_KEY=os.environ.get("HELICONE_API_KEY", None)

SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", None)