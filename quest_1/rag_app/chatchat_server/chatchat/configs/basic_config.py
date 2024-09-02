import logging
import os
from pathlib import Path

# Whether to display detailed logs
log_verbose = False

# Usually, the following content does not need to be changed

# User data root directory
DATA_PATH = str(Path(__file__).absolute().parent.parent / "data")
print(f'DATA_PATH: {DATA_PATH}')
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

# NLTK model storage path
# NLTK_DATA_PATH = os.path.join(DATA_PATH, "nltk_data")
# import nltk
# nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Log format
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

# Log storage path
LOG_PATH = os.path.join(DATA_PATH, "logs")
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

# Location to save generated content (images, videos, audios, etc.)
MEDIA_PATH = os.path.join(DATA_PATH, "media")
if not os.path.exists(MEDIA_PATH):
    os.mkdir(MEDIA_PATH)
    os.mkdir(os.path.join(MEDIA_PATH, "image"))
    os.mkdir(os.path.join(MEDIA_PATH, "audio"))
    os.mkdir(os.path.join(MEDIA_PATH, "video"))

# Temporary file directory, mainly used for file dialogs
BASE_TEMP_DIR = os.path.join(DATA_PATH, "temp")
if not os.path.exists(BASE_TEMP_DIR):
    os.mkdir(BASE_TEMP_DIR)
