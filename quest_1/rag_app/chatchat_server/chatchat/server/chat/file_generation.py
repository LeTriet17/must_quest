from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from server.utils import wrap_done, get_OpenAI, BaseResponse
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, Optional
import asyncio
from langchain_core.prompts import PromptTemplate
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, HTTPException
from server.utils import BaseResponse
from server.chat.utils import send_requests_concurrently, pdf_to_base64
from typing import Any
from langchain_core.documents import Document
import requests
from configs import LLAMA_API_KEY, MODEL_PLATFORMS
from llama_parse import LlamaParse
import aiofiles
import nest_asyncio

nest_asyncio.apply()


async def file_generation(
    file: UploadFile = File(...),
    prompt_name: str = Body("slide_generation", description="Prompt Name"),
    model_name: str = Body("gpt-4o", description="Model Name"),
):
    async with aiofiles.tempfile.NamedTemporaryFile(
        "wb", delete=False, suffix=".pdf"
    ) as temp:
        try:
            contents = await file.read()
            await temp.write(contents)
        except Exception as e:
            return HTTPException(status_code=400, detail=str(e))
        finally:
            await file.close()

    file_path = temp.name

    parser_gpt4o = LlamaParse(
        result_type=["json"],
        api_key=LLAMA_API_KEY,
        gpt4o_mode=True,
        gpt4o_api_key=MODEL_PLATFORMS[0]["api_key"],
    )
    document_pages = await parser_gpt4o.get_json_result(file_path)

    pages = [document_page['md'] for document_page in document_pages[0]['pages']]
    return BaseResponse(data=pages)
    # documents_description = send_requests_concurrently(
    #     document_pages[0]['pages'], MODEL_PLATFORMS[0]["api_key"], prompt_name, model_name
    # )
    # return BaseResponse(data=documents_description)
