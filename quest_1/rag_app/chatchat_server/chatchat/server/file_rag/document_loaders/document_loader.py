
from typing import Any
from langchain_core.documents import Document
import requests
from configs import LLAMA_API_KEY, MODEL_PLATFORMS
from llama_parse import LlamaParse
from chatchat.server.utils import get_prompt_template

class LLamaParseLoader:
     def __init__(self, *args: Any, **kwds: Any) -> None:
          print("Initializing LLamaParseLoader")
          self.file_path = args[0]
          parsing_instruction = get_prompt_template("llama_parse", "parse")
          self.parser_gpt4o = LlamaParse(
               result_type=["markdown", "json"],
               api_key=LLAMA_API_KEY,
               gpt4o_mode=True,
               gpt4o_api_key=MODEL_PLATFORMS[0]["api_key"],
               parsing_instruction=parsing_instruction,
          )

     def load(self, *args: Any, **kwds: Any) -> Document:
          self.documents_gpt4o = self.parser_gpt4o.load_data(self.file_path)
          return self.documents_gpt4o