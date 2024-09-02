from functools import lru_cache
from chatchat.server.pydantic_v2 import BaseModel, Field
from langchain.prompts.chat import ChatMessagePromptTemplate
from chatchat.configs import logger, log_verbose
from typing import List, Tuple, Dict, Union

import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from configs import MAX_TOKENS, TEMPERATURE, ENTITY_RANGE, ITERATIONS, CONTENT_CATEGORY

def create_message_template(page_elements, prompt_template):

     prompt_template.format(content_category=CONTENT_CATEGORY, entity_range=ENTITY_RANGE, max_words=MAX_TOKENS, iterations=ITERATIONS)
     messages = []
     page_text = page_elements['md']

     # Add the prompt template first
     messages.append({
          "role": "system",
          "content": [
               {
                    "type": "text",
                    "text": prompt_template
               }
          ]
     }
     )

     # Add the text content
     messages.append({
          "role": "user",
          "content": [
               {
                    "type": "text",
                    "text": page_text
               }
          ]
     }
     )
     return {"messages": messages}

# Function to send a POST request using the encoded image
def gen_page_description(page_elements, headers, prompt_template="", page_idx=0, model_name="gpt-4o"):

     try:
          payload = {
               "model": model_name,
               "max_tokens": MAX_TOKENS,
               "temperature": TEMPERATURE,
          }
          payload.update(create_message_template(page_elements, prompt_template))
          response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
          return response.status_code, response.json(), page_idx
     except Exception as e:
          return None, str(e), -1

# Use ThreadPoolExecutor to send requests concurrently
def send_requests_concurrently(document_info, api_key, prompt_name, model_name):
     results = []
     # Define headers for the API request
     headers = {
     "Content-Type": "application/json",
     "Authorization": f"Bearer {api_key}"
     }
     with ThreadPoolExecutor(max_workers=len(document_info)) as executor:
          # Start the POST requests
          future_to_key = {executor.submit(gen_page_description, page_elements, headers, prompt_name, page_idx, model_name): page_elements for page_idx, page_elements in enumerate(document_info)}
          for future in as_completed(future_to_key):
               try:
                    result = future.result()
                    results.append(result)
               except Exception as e:
                    print(f"An error occurred: {e}")
                    results.append((None, str(e)))
     # Sort the results by page index
     results.sort(key=lambda x: x[2])

     # Format the results
     responses = []
     for i, result in enumerate(results):
          responses.append({i: result[1]['choices'][0]['message']['content']})
     return responses

def pdf_to_base64(file):
    file_bytes = base64.b64encode(file.read())
    base_64 = file_bytes.decode("ascii")
    return base_64
class History(BaseModel):
    """
    Conversation history
    Can be generated from dict, e.g.,
    h = History(**{"role":"user","content":"Hello"})
    or converted to tuple, e.g.,
    h.to_msy_tuple = ("human", "Hello")
    """
    role: str = Field(...)
    content: str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role=="assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw: # Currently, default historical messages are all text without input_variable.
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        if isinstance(h, (list,tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)

        return h
