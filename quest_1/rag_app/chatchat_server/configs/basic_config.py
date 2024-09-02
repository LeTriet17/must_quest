import logging
import os
from pathlib import Path

# Whether to display detailed logs
log_verbose = False

# Usually, the following content does not need to be changed
# User data root directory
DATA_PATH = str(Path(__file__).absolute().parent.parent / "data")

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH, exist_ok=True)

API_DEPLOYMENT_NAME = "api"
WEBUI_DEPLOYMENT_NAME = "webui"
# Log format
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

# Log storage path
LOG_PATH = os.path.join(DATA_PATH, "logs")
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH, exist_ok=True)

# Temporary file directory, mainly used for file dialogs
BASE_TEMP_DIR = os.path.join(DATA_PATH, "temp")
if not os.path.exists(BASE_TEMP_DIR):
    os.makedirs(BASE_TEMP_DIR, exist_ok=True)