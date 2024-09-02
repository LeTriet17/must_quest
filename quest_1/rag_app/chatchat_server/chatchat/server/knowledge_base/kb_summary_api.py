from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from chatchat.server.utils import get_ChatOpenAI
from chatchat.server.utils import get_prompt_template
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from configs import logger
from langchain.globals import set_verbose
set_verbose(True)

def summarize_text(
    text: str,
    prompt_template: str = "default",
):
    """
    Summarizes a table based on a provided template and document.

    Args:
        docs: The Document containing the table to summarize.
        prompt_template: The template to guide the summarization (e.g., "Summarize the following table:").
        model_name: The OpenAI model to use for summarization. (gpt-3.5-turbo-16k is good for cost and performance)
        temperature: Controls the randomness of the model's output (lower is more deterministic).
        max_tokens: The maximum number of tokens allowed in the model's response.

    Returns:
        The summarized table content as a BaseResponse.
    """
    
    # Initialize the LLM
    llm = get_ChatOpenAI(seed=42)

    # Construct the prompt using the template and document content
    prompt_template = get_prompt_template("table_summary", prompt_template)
    prompt = PromptTemplate.from_template(prompt_template)
    # Load the summarization chain
    chain = LLMChain(prompt=prompt, llm=llm)
    logger.info("Summarizing text")
    # Generate the summary
    summary = chain.invoke({"context": text})
    logger.info("Summary generated")
    return summary['text']