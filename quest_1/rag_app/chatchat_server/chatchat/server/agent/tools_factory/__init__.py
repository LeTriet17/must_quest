from urllib.parse import urlencode
from chatchat.server.utils import get_tool_config
from chatchat.server.pydantic_v1 import Field
from .tools_registry import regist_tool, BaseToolOutput, ChatMode
from chatchat.server.knowledge_base.kb_api import list_kbs
from chatchat.server.knowledge_base.kb_doc_api import search_docs, DocumentWithVSId
from chatchat.configs import KB_INFO


template = "Use local knowledgebase from one or more of these:\n{KB_info}\n to get information, Only local data on this knowledge use this tool. The 'database' should be one of the above [{key}]."
KB_info_str = '\n'.join([f"{key}: {value}" for key, value in KB_INFO.items()])
template_knowledge = template.format(KB_info=KB_info_str, key="samples")
class KBToolOutput(BaseToolOutput):
    def __str__(self) -> str:
        context = ""
        docs = self.data
        source_documents = []

        for inum, doc in enumerate(docs):
            doc = DocumentWithVSId.parse_obj(doc)
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": self.extras.get("database"), "file_name": filename})
            url = f"download_doc?" + parameters
            text = f"""Source [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:
            context = "No relevant documents found, please change the keywords and try again"
        else:
            for doc in source_documents:
                context += doc + "\n"

        return context


def search_knowledgebase(query: str, database: str, config: dict):
    docs = search_docs(
        query=query,
        knowledge_base_name=database,
        top_k=config["top_k"],
        score_threshold=config["score_threshold"])
    return docs

@regist_tool(description=template_knowledge, title="Local Knowledgebase Search")
def search_local_knowledgebase(
    database: str = Field(description="Database for Knowledge Search", choices=list_kbs().data),
    query: str = Field(description="Query for Knowledge Search"),
    # chat_mode: str = Field(description="Chat or Query mode", default=ChatMode.query, choices=[ChatMode.query, ChatMode.chat])
):
    """
    Search the local knowledgebase for information
    """
    chat_mode = ChatMode.query
    tool_config = get_tool_config("search_local_knowledgebase")
    ret = search_knowledgebase(query=query, database=database, config=tool_config)
    return KBToolOutput(ret, database=database, chat_mode=chat_mode)
