from __future__ import annotations

from typing import List, Dict

from fastapi import APIRouter, Request
from langchain.prompts.prompt import PromptTemplate

from chatchat.server.api_server.api_schemas import OpenAIChatInput, MsgType, AgentStatus
from chatchat.server.chat.chat import chat, create_message
from chatchat.server.chat.feedback import chat_feedback
from chatchat.server.chat.file_chat import file_chat
from chatchat.server.chat.file_generation import file_generation
from chatchat.server.db.repository import add_message_to_db
from chatchat.server.utils import get_OpenAIClient, get_tool, get_tool_config, get_prompt_template
from .openai_routes import openai_request
from configs import logger
from sse_starlette.sse import EventSourceResponse
from chatchat.server.agent.tools_factory.tools_registry import ChatMode
chat_router = APIRouter(prefix="/chat", tags=["ChatChat"])

chat_router.post("/chat",
            summary="Talk to llm model (via LLMChain)",
            )(chat)

chat_router.post("/feedback",
            summary="Return llm model dialogue score",
            )(chat_feedback)

chat_router.post("/file_chat",
            summary="File conversation"
            )(file_chat)

chat_router.post("/chat/file_generation",
            summary="File generation"
            )(file_generation)


@chat_router.post("/chat/completions", summary="Talk to llm model (via OpenAI)")
async def chat_completions(
    request: Request,
    body: OpenAIChatInput,
) -> Dict:
    '''
    The request parameters are consistent with openai.chat.completions.create, and additional parameters can be passed in through extra_body
    Tools and tool_choice can directly pass the tool name, which will be converted according to the tools included in the project.
    Different chat functions are called with different parameter combinations:
    - tool_choice
        - extra_body contains tool_input: directly call tool_choice(tool_input)
        - extra_body does not contain tool_input: call tool_choice through agent
    - tools: agent dialogue
    - Others: LLM dialogue
    Other combinations (such as file conversations) will also be considered in the future.
    Returns a Dict compatible with openai
    '''
    client = get_OpenAIClient(model_name=body.model, is_async=True)
    extra = {**body.model_extra} or {}
    for key in list(extra):
        delattr(body, key)

    # check tools & tool_choice in request body
    if isinstance(body.tool_choice, str):
        if t := get_tool(body.tool_choice):
            body.tool_choice = {"function": {"name": t.name}, "type": "function"}
    if isinstance(body.tools, list):
        for i in range(len(body.tools)):
            if isinstance(body.tools[i], str):
                if t := get_tool(body.tools[i]):
                    body.tools[i] = {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.args,
                        }
                    }

    conversation_id = extra.get("conversation_id")

    # chat based on result from one choiced tool
    if body.tool_choice:
        print(f'Chat with tool_choice: {body.tool_choice["function"]["name"]}')
        tool = get_tool(body.tool_choice["function"]["name"])
        if not body.tools:
            body.tools = [{
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.args,
                        }
                    }]
        if tool_input := extra.get("tool_input"):
            message_id = add_message_to_db(
                chat_type="tool_call",
                query=body.messages[-1]["content"],
                conversation_id=conversation_id
            ) if conversation_id else None
            

            tool_result = await tool.ainvoke(tool_input)
            tool_result_str = str(tool_result)
            if body.tool_choice["function"]["name"] == "search_local_knowledgebase":
                chat_mode = tool_result.chat_mode
                if chat_mode==ChatMode.query and tool_result_str.startswith("No relevant documents found"):
                    logger.info(f'No relevant documents found')
                    return await create_message(query="Sorry, I couldn't find any relevant documents for your question. Please try asking in a different way.",
                            metadata=extra.get("metadata", {}),
                            conversation_id=extra.get("conversation_id", ""),
                            message_id=message_id,
                            )
            if chat_mode==ChatMode.query:
                prompt_template = PromptTemplate.from_template(get_prompt_template("llm_model", "rag_query"))
            elif chat_mode==ChatMode.chat:
                prompt_template = PromptTemplate.from_template(get_prompt_template("llm_model", "rag_chat"))
            body.messages[-1]["content"] = prompt_template.format(context=tool_result, question=body.messages[-1]["content"])
            del body.tools
            del body.tool_choice
            extra_json = {
                "message_id": message_id,
                "status": None,
            }
            header = [{**extra_json,
                       "content": f"{tool_result}",
                       "tool_output":tool_result.data,
                       "is_ref": True,
                       }]
            return await openai_request(client.chat.completions.create, body, extra_json=extra_json, header=header)

    # agent chat with tool calls
    if body.tools:
        message_id = add_message_to_db(
            chat_type="agent_chat",
            query=body.messages[-1]["content"],
            conversation_id=conversation_id
        ) if conversation_id else None

        chat_model_config = {} # TODO: 前端支持配置模型
        tool_names = [x["function"]["name"] for x in body.tools]
        tool_config = {name: get_tool_config(name) for name in tool_names}
        result = await chat(query=body.messages[-1]["content"],
                        metadata=extra.get("metadata", {}),
                        conversation_id=extra.get("conversation_id", ""),
                        message_id=message_id,
                        history_len=-1,
                        history=body.messages[:-1],
                        stream=body.stream,
                        chat_model_config=extra.get("chat_model_config", chat_model_config),
                        tool_config=extra.get("tool_config", tool_config),
                        )
        return result
    else: # LLM chat directly
        message_id = add_message_to_db(
            chat_type="llm_chat",
            query=body.messages[-1]["content"],
            conversation_id=conversation_id
        ) if conversation_id else None
        extra_json = {
            "message_id": message_id,
            "status": None,
        }
        return await openai_request(client.chat.completions.create, body, extra_json=extra_json)
