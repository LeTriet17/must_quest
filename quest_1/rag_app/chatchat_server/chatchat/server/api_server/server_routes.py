from typing import Literal

from fastapi import APIRouter, Body

from chatchat.server.utils import get_server_configs, get_prompt_template


server_router = APIRouter(prefix="/server", tags=["Server State"])


# Server-related endpoints
server_router.post("/configs",
            summary="Get raw server configuration information",
            )(get_server_configs)


@server_router.post("/get_prompt_template",
            summary="Get the prompt template configured for the server")
def get_server_prompt_template(
    type: Literal["llm_chat", "knowledge_base_chat"]=Body("llm_chat", description="Template type, options: llm_chat, knowledge_base_chat"),
    name: str = Body("default", description="Template name"),
) -> str:
    return get_prompt_template(type=type, name=name)
