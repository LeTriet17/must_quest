import asyncio
from contextlib import asynccontextmanager
import multiprocessing as mp
import os
import subprocess
import sys
from multiprocessing import Process
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import numexpr

    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

from chatchat.configs import (
    LOG_PATH,
    log_verbose,
    logger,
    DEFAULT_EMBEDDING_MODEL,
    TEXT_SPLITTER_NAME,
    API_SERVER,
    WEBUI_SERVER,
)
from chatchat.server.utils import FastAPI
from chatchat.server.knowledge_base.migrate import create_tables
import argparse
from typing import List, Dict
from chatchat.configs import VERSION


def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if started_event is not None:
            started_event.set()
        yield
    app.router.lifespan_context = lifespan


def run_api_server(started_event: mp.Event = None, run_mode: str = None):
    from chatchat.server.api_server.server_app import create_app
    import uvicorn
    from chatchat.server.utils import set_httpx_config
    set_httpx_config()

    app = create_app(run_mode=run_mode)
    _set_app_event(app, started_event)

    host = API_SERVER["host"]
    port = API_SERVER["port"]

    uvicorn.run(app, host=host, port=port, loop='asyncio')


def run_webui(started_event: mp.Event = None, run_mode: str = None):
    import sys
    from chatchat.server.utils import set_httpx_config

    set_httpx_config()

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]

    # st_exe = os.path.join(os.path.dirname(sys.executable),"scripts","streamlit")
    st_exe = "streamlit"
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webui.py')
    cmd = [st_exe, "run", script_dir,
        "--client.showErrorDetails=false",
        "--server.address", host,
        "--server.port", str(port),
        "--theme.base", "dark",
        "--theme.primaryColor", "#486581",
        "--theme.secondaryBackgroundColor", "#333333",  # Example secondary background color for dark theme
        "--theme.textColor", "#ffffff",  # White text for dark theme
        "--browser.gatherUsageStats", "false",
        ]
    if run_mode == "lite":
        cmd += [
            "--",
            "lite",
        ]
    p = subprocess.Popen(cmd)
    started_event.set()
    p.wait()


def run_loom(started_event: mp.Event = None):
    from chatchat.configs import LOOM_CONFIG

    cmd = ["python", "-m", "loom_core.openai_plugins.deploy.local",
           "-f", LOOM_CONFIG
           ]

    p = subprocess.Popen(cmd)
    started_event.set()
    p.wait()


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--all-webui",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py and webui.py",
        dest="all_webui",
    )
    parser.add_argument(
        "--all-api",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py",
        dest="all_api",
    )

    parser.add_argument(
        "--api",
        action="store_true",
        help="run api.py server",
        dest="api",
    )

    parser.add_argument(
        "-w",
        "--webui",
        action="store_true",
        help="run webui.py server",
        dest="webui",
    )
    parser.add_argument(
        "-i",
        "--lite",
        action="store_true",
        help="run webui.py server in lite mode",
        dest="lite",
    )
    args = parser.parse_args()
    return args, parser


def dump_server_info(after_start=True, args=None):
    import platform
    import langchain
    from chatchat.server.utils import api_address, webui_address

    print("\n")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print(f"Operating System: {platform.platform()}.")
    print(f"Python Version: {sys.version}")
    print(f"Project Version: {VERSION}")
    print(f"Langchain Version: {langchain.__version__}")
    print("\n")

    print(f"Currently Used Tokenizer: {TEXT_SPLITTER_NAME}")

    print(f"Current Embeddings Model: {DEFAULT_EMBEDDING_MODEL}")

    if after_start:
        print("\n")
        print(f"Server Running Information:")
        if args.webui:
            print(f"    Chatchat WEBUI Server: {webui_address()}")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print("\n")

async def start_main_server():
    import time
    import signal

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()
    run_mode = None

    args, parser = parse_args()

    if args.all_webui:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = True

    elif args.all_api:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = False

    if args.lite:
        args.model_worker = False
        run_mode = "lite"

    dump_server_info(args=args)

    if len(sys.argv) > 1:
        logger.info(f"Starting service:")
        logger.info(f"To view the llm_api log, please go to {LOG_PATH}")

    processes = {"online_api": {}, "model_worker": {}}

    def process_count():
        return len(processes)

    loom_started = manager.Event()
    # process = Process(
    #     target=run_loom,
    #     name=f"run_loom Server",
    #     kwargs=dict(started_event=loom_started),
    #     daemon=True,
    # )
    # processes["run_loom"] = process
    api_started = manager.Event()
    if args.api:
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(started_event=api_started, run_mode=run_mode),
            daemon=True,
        )
        processes["api"] = process

    webui_started = manager.Event()
    if args.webui:
        process = Process(
            target=run_webui,
            name=f"WEBUI Server",
            kwargs=dict(started_event=webui_started, run_mode=run_mode),
            daemon=True,
        )
        processes["webui"] = process

    if process_count() == 0:
        parser.print_help()
    else:
        try:
            # Ensure that the loom server is started first
            if p := processes.get("run_loom"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                loom_started.wait()  #  Wait for loom server to start

            if p := processes.get("api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                api_started.wait()  # Wait for api.py to start

            if p := processes.get("webui"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                webui_started.wait()  # Wait for webui.py to start

            dump_server_info(after_start=True, args=args)

            if p := processes.get("api"):
                p.join()
            if p := processes.get("webui"):
                p.join()
        except Exception as e:
            logger.error(e)
            logger.warning("Caught KeyboardInterrupt! Setting stop event...")
        finally:

            for p in processes.values():
                logger.warning("Sending SIGKILL to %s", p)
                # Queues and other inter-process communication primitives can break when
                # process is killed, but we don't care here

                if isinstance(p, dict):
                    for process in p.values():
                        process.kill()
                else:
                    p.kill()

            for p in processes.values():
                logger.info("Process status: %s", p)


if __name__ == "__main__":
    print("Starting server")
    load_dotenv() 
    create_tables()
    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
    loop.run_until_complete(start_main_server())
