from fastapi import FastAPI
from pathlib import Path
import asyncio
import os
import sys
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from langchain_core.embeddings import Embeddings
from langchain.tools import BaseTool
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
import httpx
import openai
from typing import (
    Optional,
    Callable,
    Generator,
    Dict,
    List,
    Any,
    Awaitable,
    Union,
    Tuple,
    Literal,
)
import logging

from chatchat.configs import (logger, log_verbose, HTTPX_DEFAULT_TIMEOUT,
                     DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, TEMPERATURE)
from chatchat.server.pydantic_v2 import BaseModel, Field


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        logging.exception(e)
        msg = f"Caught exception: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
    finally:
        # Signal the aiter to stop.
        event.set()


def get_config_platforms() -> Dict[str, Dict]:
    import importlib
    from chatchat.configs import model_config
    importlib.reload(model_config)

    return {m["platform_name"]: m for m in model_config.MODEL_PLATFORMS}


def get_config_models(
    model_name: str = None,
    model_type: Literal["llm", "embed", "image", "multimodal"] = None,
    platform_name: str = None,
) -> Dict[str, Dict]:
    '''
    Get the configured model list, the return value is:
    {model_name: {
        "platform_name": xx,
        "platform_type": xx,
        "model_type": xx,
        "model_name": xx,
        "api_base_url": xx,
        "api_key": xx,
        "api_proxy": xx,
    }}
    '''
    import importlib
    from chatchat.configs import model_config
    importlib.reload(model_config)

    result = {}
    for m in model_config.MODEL_PLATFORMS:
        if platform_name is not None and platform_name != m.get("platform_name"):
            continue
        if model_type is not None and f"{model_type}_models" not in m:
            continue

        if model_type is None:
            model_types = ["llm_models", "embed_models", "image_models", "multimodal_models"]
        else:
            model_types = [f"{model_type}_models"]

        for m_type in model_types:
            for m_name in m.get(m_type, []):
                if model_name is None or model_name == m_name:
                    result[m_name] = {
                        "platform_name": m.get("platform_name"),
                        "platform_type": m.get("platform_type"),
                        "model_type": m_type.split("_")[0],
                        "model_name": m_name,
                        "api_base_url": m.get("api_base_url"),
                        "api_key": m.get("api_key"),
                        "api_proxy": m.get("api_proxy"),
                    }
    return result


def get_model_info(model_name: str = None, platform_name: str = None, multiple: bool = False) -> Dict:
    '''
    Get configured model information, mainly api_base_url and api_key.
    If multiple=True is specified, returns all models with the same name; otherwise, only returns the first one.
    '''
    result = get_config_models(model_name=model_name, platform_name=platform_name)
    if len(result) > 0:
        if multiple:
            return result
        else:
            return list(result.values())[0]
    else:
        return {}


def get_ChatOpenAI(
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = None,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        local_wrap: bool = False, # use local wrapped api
        **kwargs: Any,
) -> ChatOpenAI:
    model_info = get_model_info(model_name)
    params = dict(
            streaming=streaming,
            verbose=verbose,
            callbacks=callbacks,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
    )
    try:
        if local_wrap:
            params.update(
                openai_api_base = f"{api_address()}/v1",
                openai_api_key = "EMPTY",
            )
        else:
            params.update(
                openai_api_base=model_info.get("api_base_url"),
                openai_api_key=model_info.get("api_key"),
                openai_proxy=model_info.get("api_proxy"),
            )
        model = ChatOpenAI(**params)
    except Exception as e:
        logger.error(f"failed to create ChatOpenAI for model: {model_name}.", exc_info=True)
        model = None
    return model


def get_OpenAI(
        model_name: str,
        temperature: float,
        max_tokens: int = None,
        streaming: bool = True,
        echo: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        local_wrap: bool = False, # use local wrapped api
        **kwargs: Any,
) -> OpenAI:
    model_info = get_model_info(model_name)
    params = dict(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        echo=echo,
        **kwargs
    )
    try:
        if local_wrap:
            params.update(
                openai_api_base = f"{api_address()}/v1",
                openai_api_key = "EMPTY",
            )
        else:
            params.update(
                openai_api_base=model_info.get("api_base_url"),
                openai_api_key=model_info.get("api_key"),
                openai_proxy=model_info.get("api_proxy"),
            )
        model = OpenAI(**params)
    except Exception as e:
        logger.error(f"failed to create OpenAI for model: {model_name}.", exc_info=True)
        model = None
    return model


def get_Embeddings(
    embed_model: str = DEFAULT_EMBEDDING_MODEL,
    local_wrap: bool = False, # use local wrapped api
) -> Embeddings:
    from langchain_community.embeddings.openai import OpenAIEmbeddings
    from langchain_community.embeddings import OllamaEmbeddings
    from chatchat.server.localai_embeddings import LocalAIEmbeddings # TODO: fork of lc pr #17154

    model_info = get_model_info(model_name=embed_model)
    params = dict(model=embed_model)
    try:
        if local_wrap:
            params.update(
                openai_api_base = f"{api_address()}/v1",
                openai_api_key = "EMPTY",
            )
        else:
            params.update(
                openai_api_base=model_info.get("api_base_url"),
                openai_api_key=model_info.get("api_key"),
                openai_proxy=model_info.get("api_proxy"),
            )
        if model_info.get("platform_type") == "openai":
            return OpenAIEmbeddings(**params)
        elif model_info.get("platform_type") == "ollama":
            return OllamaEmbeddings(base_url=model_info.get("api_base_url").replace('/v1',''),
                                    model=embed_model,
                                    )
        else:
            return LocalAIEmbeddings(**params)
    except Exception as e:
        logger.error(f"failed to create Embeddings for model: {embed_model}.", exc_info=True)


def get_OpenAIClient(
        platform_name: str=None,
        model_name: str=None,
        is_async: bool=True,
) -> Union[openai.Client, openai.AsyncClient]:
    '''
    construct an openai Client for specified platform or model
    '''
    if platform_name is None:
        platform_name = get_model_info(model_name=model_name, platform_name=platform_name)["platform_name"]
    platform_info = get_config_platforms().get(platform_name)
    assert platform_info, f"cannot find configured platform: {platform_name}"
    params = {
        "base_url": platform_info.get("api_base_url"),
        "api_key": platform_info.get("api_key"),
    }
    httpx_params = {}
    if api_proxy := platform_info.get("api_proxy"):
        httpx_params = {
            "proxies": api_proxy,
            "transport": httpx.HTTPTransport(local_address="0.0.0.0"),
        }

    if is_async:
        if httpx_params:
            params["http_client"] = httpx.AsyncClient(**httpx_params)
        return openai.AsyncClient(**params)
    else:
        if httpx_params:
            params["http_client"] = httpx.Client(**httpx_params)
        return openai.Client(**params)


class MsgType:
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4


class BaseResponse(BaseModel):
    code: int = Field(200, description="API status code")
    msg: str = Field("success", description="API status message")
    data: Any = Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListResponse(BaseResponse):
    data: List[str] = Field(..., description="List of names")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    question: str = Field(..., description="Question text")
    response: str = Field(..., description="Response text")
    history: List[List[str]] = Field(..., description="History text")
    source_documents: List[str] = Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How to handle work injury insurance?",
                "response": "Based on the known information, it can be summarized as follows:\n\n1. The employing unit pays work injury insurance premiums for employees to ensure that they receive corresponding benefits in the event of a work injury.\n"
                            "2. The payment regulations for work injury insurance may vary in different regions. It is necessary to consult the local social security department to understand the specific payment standards and regulations.\n"
                            "3. Employees and their immediate family members involved in work injuries need to apply for work injury identification, confirm their eligibility for benefits, and pay work injury insurance premiums on time.\n"
                            "4. Work injury insurance benefits include medical treatment for work-related injuries, rehabilitation, assistive device costs, disability benefits, death benefits, and one-time death subsidies, among others.\n"
                            "5. Qualifications for receiving work injury insurance benefits include certification for long-term benefit recipients and one-time benefit recipients.\n"
                            "6. Benefits paid by the work injury insurance fund include work injury medical benefits, rehabilitation benefits, assistive device costs, one-time death subsidies, funeral subsidies, and more.",
                "history": [
                    [
                        "What is work injury insurance?",
                        "Work injury insurance refers to a social insurance system in which employers pay work injury insurance premiums for their employees and other personnel of the employing unit according to national regulations. Insurance institutions provide work injury insurance benefits according to national standards.",
                    ]
                ],
                "source_documents": [
                    "Source [1] Guidelines for Employers in Guangzhou to Participate in Work Injury Insurance for Specific Personnel.docx:\n\n\t"
                    "(1) Employers (organizations) adopt the principle of 'voluntary participation' to allow specific employees without established labor relations to participate in work injury insurance and pay work injury insurance premiums separately.",
                    "Source [2] ...",
                    "Source [3] ...",
                ],
            }
        }



def run_async(cor):
    '''
    Run an async function in the event loop.
    '''
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
    return loop.run_until_complete(cor)


def iter_over_async(ait, loop=None):
    '''
    将异步生成器封装成同步生成器.
    '''
    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj


def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parent / "api_server" / "static",
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse

    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        '''
        remove original route from app
        '''
        index = None
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            app.routes.pop(index)

    # Set up static file mount
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    if docs_url is not None:
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    if redoc_url is not None:
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )


# 从model_config中获取模型信息
# TODO: 移出模型加载后，这些功能需要删除或改变实现

# def list_embed_models() -> List[str]:
#     '''
#     get names of configured embedding models
#     '''
#     return list(MODEL_PATH["embed_model"])


# def get_model_path(model_name: str, type: str = None) -> Optional[str]:
#     if type in MODEL_PATH:
#         paths = MODEL_PATH[type]
#     else:
#         paths = {}
#         for v in MODEL_PATH.values():
#             paths.update(v)

#     if path_str := paths.get(model_name):  # 以 "chatglm-6b": "THUDM/chatglm-6b-new" 为例，以下都是支持的路径
#         path = Path(path_str)
#         if path.is_dir():  # 任意绝对路径
#             return str(path)

#         root_path = Path(MODEL_ROOT_PATH)
#         if root_path.is_dir():
#             path = root_path / model_name
#             if path.is_dir():  # use key, {MODEL_ROOT_PATH}/chatglm-6b
#                 return str(path)
#             path = root_path / path_str
#             if path.is_dir():  # use value, {MODEL_ROOT_PATH}/THUDM/chatglm-6b-new
#                 return str(path)
#             path = root_path / path_str.split("/")[-1]
#             if path.is_dir():  # use value split by "/", {MODEL_ROOT_PATH}/chatglm-6b-new
#                 return str(path)
#         return path_str  # THUDM/chatglm06b


def api_address() -> str:
    from chatchat.configs.server_config import API_SERVER
    app_env = os.environ.get("APP_ENV", "dev")
    if app_env == "dev":
        host = API_SERVER["host"]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = API_SERVER["port"]
        print(f"http://{host}:{port}")
        return f"http://{host}:{port}"
    elif app_env == "prod":
        host = API_SERVER["host_prod"]
        return f"https://{host}/api"
    else:
        # Raise error
        raise ValueError(f"Unknown APP_ENV: {app_env}, should be 'dev' or 'prod'.")


def webui_address() -> str:
    from chatchat.configs.server_config import WEBUI_SERVER

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]
    return f"http://{host}:{port}"


def get_prompt_template(type: str, name: str) -> Optional[str]:
    '''
    Get the prompt template from the prompt_config.py file.
    '''

    from chatchat.configs import prompt_config
    import importlib
    importlib.reload(prompt_config)  # reload the module to get the latest changes
    return prompt_config.PROMPT_TEMPLATES.get(type, {}).get(name)

def set_httpx_config(
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        proxy: Union[str, Dict] = None,
        unused_proxies: List[str] = [],
):
    '''
    Set the default timeout for httpx. The default timeout for httpx is 5 seconds, which is not enough when requesting LLM responses.
    Add relevant services of this project to the list of unused proxies to avoid request errors from fastchat's servers. (Not effective on Windows)
    For online APIs like ChatGPT, if you want to use a proxy, you need to configure it manually. Considerations are also needed for how to handle proxies for search engines.
    '''

    import httpx
    import os

    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # Set the default proxy for httpx
    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    for k, v in proxies.items():
        os.environ[k] = v

    # set host to bypass proxy
    no_proxy = [x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()]
    no_proxy += [
        # do not use proxy for locahost
        "http://127.0.0.1",
        "http://localhost",
    ]
    # do not use proxy for user deployed fastchat servers
    for x in unused_proxies:
        host = ":".join(x.split(":")[:2])
        if host not in no_proxy:
            no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    def _get_proxies():
        return proxies

    import urllib.request
    urllib.request.getproxies = _get_proxies


def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    '''
    Run tasks in a thread pool and return the results as a generator.
    '''
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            tasks.append(pool.submit(func, **kwargs))

        for obj in as_completed(tasks):
            try:
                yield obj.result()
            except Exception as e:
                logger.error(f"error in sub thread: {e}", exc_info=True)


def run_in_process_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    '''
    Run tasks in a process pool and return the results as a generator.
    '''
    tasks = []
    max_workers = None
    if sys.platform.startswith("win"):
        max_workers = min(mp.cpu_count(), 60) # max_workers should not exceed 60 on windows
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for kwargs in params:
            tasks.append(pool.submit(func, **kwargs))

        for obj in as_completed(tasks):
            try:
                yield obj.result()
            except Exception as e:
                logger.error(f"error in sub process: {e}", exc_info=True)


def get_httpx_client(
        use_async: bool = False,
        proxies: Union[str, Dict] = None,
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        unused_proxies: List[str] = [],
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    '''
    helper to get httpx client with default proxies that bypass local addesses.
    '''
    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # do not use proxy for user deployed fastchat servers
    for x in unused_proxies:
        host = ":".join(x.split(":")[:2])
        default_proxies.update({host: None})

    # get proxies from system envionrent
    # proxy not str empty string, None, False, 0, [] or {}
    default_proxies.update({
        "http://": (os.environ.get("http_proxy")
                    if os.environ.get("http_proxy") and len(os.environ.get("http_proxy").strip())
                    else None),
        "https://": (os.environ.get("https_proxy")
                     if os.environ.get("https_proxy") and len(os.environ.get("https_proxy").strip())
                     else None),
        "all://": (os.environ.get("all_proxy")
                   if os.environ.get("all_proxy") and len(os.environ.get("all_proxy").strip())
                   else None),
    })
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # Origin code
            default_proxies.update({'all://' + host: None})  # PR 1838 fix, if not add 'all://', httpx will raise error

    # merge default proxies with user provided proxies
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # construct Client
    kwargs.update(timeout=timeout, proxies=default_proxies)

    if log_verbose:
        logger.info(f'{get_httpx_client.__class__.__name__}:kwargs: {kwargs}')

    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)


def get_server_configs() -> Dict:
    '''
    Get server configurations, including api_address.
    '''
    _custom = {
        "api_address": api_address(),
    }

    return {**{k: v for k, v in locals().items() if k[0] != "_"}, **_custom}


def get_temp_dir(id: str = None) -> Tuple[str, str]:
    '''
    Get a temporary directory for storing files.
    '''
    from chatchat.configs.basic_config import BASE_TEMP_DIR
    import uuid

    if id is not None:  # If the id is specified, use the specified id as the directory name
        path = os.path.join(BASE_TEMP_DIR, id)
        if os.path.isdir(path):
            return path, id

    id = uuid.uuid4().hex
    path = os.path.join(BASE_TEMP_DIR, id)
    os.mkdir(path)
    return path, id


def get_tool(name: str = None) -> Union[BaseTool, Dict[str, BaseTool]]:
    import importlib
    from chatchat.server.agent import tools_factory
    importlib.reload(tools_factory) # reload the module to get the latest changes
    # list_args = tools_factory.tools_registry._TOOLS_REGISTRY["search_local_knowledgebase"].args["database"]["choices"]
    if name is None:
        return tools_factory.tools_registry._TOOLS_REGISTRY
    else:
        return tools_factory.tools_registry._TOOLS_REGISTRY.get(name)


def get_tool_config(name: str = None) -> Dict:
    import importlib
    from chatchat.configs import model_config
    importlib.reload(model_config)

    if name is None:
        return model_config.TOOL_CONFIG
    else:
        return model_config.TOOL_CONFIG.get(name, {})
