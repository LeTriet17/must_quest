from chatchat.server.file_rag.retrievers.base import BaseRetrieverService
from langchain.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever


class VectorstoreRetrieverService(BaseRetrieverService):
    def do_init(
            self,
            retriever: BaseRetriever = None,
            top_k: int = 5,
    ):
        self.vs = None
        self.top_k = top_k
        self.retriever = None


    @staticmethod
    def from_vectorstore(
            vectorstore: VectorStore,
            top_k: int,
            score_threshold: int or float,
    ):  
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,
                # "k": top_k
            }
        )
        # return VectorstoreRetrieverService(retriever=retriever)
        return retriever

    def get_related_documents(self, query: str):
        self.retriever.invoke(query)[:self.top_k]
