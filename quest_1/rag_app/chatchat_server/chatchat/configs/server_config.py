import sys


# Default timeout for httpx requests (in seconds). If loading models or dialogs is slow and timeouts occur, you can increase this value appropriately.
HTTPX_DEFAULT_TIMEOUT = 1000.0

# Whether the API enables cross-origin requests. Default is False, if needed, please set it to True.
OPEN_CROSS_DOMAIN = True

# Default host binding for each server. If changed to "0.0.0.0", you need to modify the host of all XX_SERVER below.
DEFAULT_BIND_HOST = "127.0.0.1" if sys.platform != "win32" else "127.0.0.1"


# webui.py server
WEBUI_SERVER = {
    "host": '0.0.0.0',
    "port": 8501,
}

# api.py server
API_SERVER = {
    "host": '0.0.0.0',
    "port": 7861,
    "host_prod": "demo-odd-api.capq.ai"
}
