import argparse
from typing import Literal

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
import uvicorn

from chatchat.configs import VERSION, MEDIA_PATH
from chatchat.configs.server_config import OPEN_CROSS_DOMAIN
from chatchat.server.api_server.chat_routes import chat_router
from chatchat.server.api_server.kb_routes import kb_router
from chatchat.server.api_server.openai_routes import openai_router
from chatchat.server.api_server.server_routes import server_router
from chatchat.server.api_server.tool_routes import tool_router
from chatchat.server.chat.completion import completion
from chatchat.server.utils import MakeFastAPIOffline


def create_app(run_mode: str=None):
    app = FastAPI(
        title="Langchain-Chatchat API Server",
        version=VERSION,
        root_path="/api"
    )

    MakeFastAPIOffline(app)
    # Add CORS middleware to allow all origins
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get('/health', include_in_schema=False)
    async def health():
        return {"status": "ok"}

    @app.get("/", summary="swagger documentation", include_in_schema=False)
    async def document():
        return RedirectResponse(url="/docs")

    app.include_router(chat_router)
    app.include_router(kb_router)
    app.include_router(tool_router)
    app.include_router(openai_router)
    app.include_router(server_router)

    # Other endpoints
    app.post("/other/completion",
             tags=["Other"],
             summary="Request completion from LLM model (via LLMChain)",
             )(completion)

    # Media files
    app.mount("/media", StaticFiles(directory=MEDIA_PATH), name="media")

    return app


def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)

app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Renn AI PDF API Server',
                                     description='API Server for Renn AI PDF',
                                     )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # Initialize arguments
    args = parser.parse_args()
    args_dict = vars(args)

    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
