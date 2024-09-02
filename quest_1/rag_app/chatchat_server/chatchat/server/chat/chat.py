import asyncio
import json
import time
from typing import AsyncIterable, List
import uuid

from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from chatchat.configs.model_config import LLM_MODEL_CONFIG
from chatchat.server.agent.agent_factory.agents_registry import agents_registry
from chatchat.server.agent.container import container
from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from chatchat.server.utils import wrap_done, get_ChatOpenAI, get_prompt_template, MsgType, get_tool
from chatchat.server.chat.utils import History
from chatchat.server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from chatchat.server.callback_handler.agent_callback_handler import AgentExecutorAsyncIteratorCallbackHandler, AgentStatus


def create_models_from_config(configs, callbacks, stream):
    configs = configs or LLM_MODEL_CONFIG
    models = {}
    prompts = {}
    for model_type, model_configs in configs.items():
        for model_name, params in model_configs.items():
            callbacks = callbacks if params.get('callbacks', False) else None
            model_instance = get_ChatOpenAI(
                model_name=model_name,
                temperature=params.get('temperature', 0.5),
                max_tokens=params.get('max_tokens', 1000),
                callbacks=callbacks,
                streaming=stream,
                local_wrap=True,
            )
            models[model_type] = model_instance
            prompt_name = params.get('prompt_name', 'default')
            prompt_template = get_prompt_template(type=model_type, name=prompt_name)
            prompts[model_type] = prompt_template
    return models, prompts


def create_models_chains(history, history_len, prompts, models, tools, callbacks, conversation_id, metadata):
    memory = None
    chat_prompt = None
    container.metadata = metadata

    if history:
        history = [History.from_data(h) for h in history]
        input_msg = History(role="user", content=prompts["llm_model"]).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
    elif conversation_id and history_len > 0:
        memory = ConversationBufferDBMemory(
            conversation_id=conversation_id,
            llm=models["llm_model"],
            message_limit=history_len
        )
    else:
        input_msg = History(role="user", content=prompts["llm_model"]).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])

    llm=models["llm_model"]
    llm.callbacks = callbacks
    chain = LLMChain(
        prompt=chat_prompt,
        llm=llm,
        memory=memory
    )
    classifier_chain = (
            PromptTemplate.from_template(prompts["preprocess_model"])
            | models["preprocess_model"]
            | StrOutputParser()
    )

    if "action_model" in models and tools is not None:
        agent_executor = agents_registry(
            llm=llm,
            callbacks=callbacks,
            tools=tools,
            prompt=None,
            verbose=True
        )
        # branch = RunnableBranch(
        #     (lambda x: "1" in x["topic"].lower(), agent_executor),
        #     chain
        # )
        # full_chain = ({"topic": classifier_chain, "input": lambda x: x["input"]} | branch)
        full_chain = ({"input": lambda x: x["input"]} | agent_executor)
    else:
        chain.llm.callbacks = callbacks
        full_chain = ({"input": lambda x: x["input"]} | chain)
    return full_chain

async def create_message(query: str = Body(..., description="User input", examples=["angry"]),
            metadata: dict = Body({}, description="Attachments, which may be images or other functions", examples=[]),
            conversation_id: str = Body("", description="Dialog ID"),
            message_id: str = Body(None, description="Database message ID"),
            ):
    '''Agent chat with tool calls'''

    async def chat_iterator() -> AsyncIterable[OpenAIChatOutput]:
        ret = OpenAIChatOutput(
                id=f"chat{uuid.uuid4()}",
                object="chat.completion.chunk",
                content=query,
                role="assistant",
                tool_calls=[],
                message_type = 1,
                message_id=message_id,
            )
        yield ret.model_dump_json()
    return EventSourceResponse(chat_iterator())
        

    

async def chat(query: str = Body(..., description="User input", examples=["angry"]),
            metadata: dict = Body({}, description="Attachments, which may be images or other functions", examples=[]),
            conversation_id: str = Body("", description="Dialog ID"),
            message_id: str = Body(None, description="Database message ID"),
            history_len: int = Body(-1, description="The number of historical messages taken from the database"),
            history: List[History] = Body(
                [],
                description="Historical conversation, set to an integer to read historical messages from the database",
                examples=[
                    [
                        {"role": "user",
                            "content": "Let's play Idiom Solitaire, I'll go first, lively and energetic"},
                        {"role": "assistant", "content": "Tiger Head and Tiger Brain"}
                    ]
                ]
            ),
            stream: bool = Body(True, description="Streaming output"),
            chat_model_config: dict = Body({}, description="LLM model configuration", examples=[]),
            tool_config: dict = Body({}, description="Tool Configuration", examples=[]),
            ):
    '''Agent chat with tool calls'''

    async def chat_iterator() -> AsyncIterable[OpenAIChatOutput]:
        callback = AgentExecutorAsyncIteratorCallbackHandler()
        callbacks = [callback]
        models, prompts = create_models_from_config(callbacks=callbacks, configs=chat_model_config,
                                                    stream=stream)
        all_tools = get_tool().values()
        tools = [tool for tool in all_tools if tool.name in tool_config]
        tools = [t.copy(update={"callbacks": callbacks}) for t in tools]
        full_chain = create_models_chains(prompts=prompts,
                                          models=models,
                                          conversation_id=conversation_id,
                                          tools=tools,
                                          callbacks=callbacks,
                                          history=history,
                                          history_len=history_len,
                                          metadata=metadata)
        task = asyncio.create_task(wrap_done(
            full_chain.ainvoke(
                {
                    "input": query,
                    "chat_history": [],
                }
            ), callback.done))

        last_tool = {}
        async for chunk in callback.aiter():
            data = json.loads(chunk)
            data["tool_calls"] = []
            data["message_type"] = MsgType.TEXT

            if data["status"] == AgentStatus.tool_start:
                last_tool = {
                    "index": 0,
                    "id": data["run_id"],
                    "type": "function",
                    "function": {
                        "name": data["tool"],
                        "arguments": data["tool_input"],
                    },
                    "tool_output": None,
                    "is_error": False,
                }
                data["tool_calls"].append(last_tool)
            if data["status"] in [AgentStatus.tool_end]:
                last_tool.update(
                    tool_output=data["tool_output"],
                    is_error=data.get("is_error", False)
                )
                data["tool_calls"] = [last_tool]
                last_tool = {}
                try:
                    tool_output = json.loads(data["tool_output"])
                    if message_type := tool_output.get("message_type"):
                        data["message_type"] = message_type
                except:
                    ...
            elif data["status"] == AgentStatus.agent_finish:
                try:
                    tool_output = json.loads(data["text"])
                    if message_type := tool_output.get("message_type"):
                        data["message_type"] = message_type
                except:
                    ...

            ret = OpenAIChatOutput(
                id=f"chat{uuid.uuid4()}",
                object="chat.completion.chunk",
                content=data.get("text", ""),
                role="assistant",
                tool_calls=data["tool_calls"],
                model=models["llm_model"].model_name,
                status = data["status"],
                message_type = data["message_type"],
                message_id=message_id,
            )
            yield ret.model_dump_json()
        await task

    if stream:
        return EventSourceResponse(chat_iterator())
    else:
        ret = OpenAIChatOutput(
            id=f"chat{uuid.uuid4()}",
            object="chat.completion",
            content="",
            role="assistant",
            finish_reason="stop",
            tool_calls=[],
            status = AgentStatus.agent_finish,
            message_type = MsgType.TEXT,
            message_id=message_id,
        )

        async for chunk in chat_iterator():
            data = json.loads(chunk)
            if text := data["choices"][0]["delta"]["content"]:
                ret.content += text
            if data["status"] == AgentStatus.tool_end:
                ret.tool_calls += data["choices"][0]["delta"]["tool_calls"]
            ret.model = data["model"]
            ret.created = data["created"]

        return ret.model_dump()
