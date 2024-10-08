from langchain.vectorstores import VectorStore
from abc import ABCMeta, abstractmethod


class BaseRetrieverService(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.do_init(**kwargs)

    @abstractmethod
    def do_init(self, **kwargs):
        pass


    @abstractmethod
    def from_vectorstore(
            vectorstore: VectorStore,
            top_k: int,
            score_threshold: int or float,
    ):
        pass

    @abstractmethod
    def get_related_documents(self, query: str):
        pass
