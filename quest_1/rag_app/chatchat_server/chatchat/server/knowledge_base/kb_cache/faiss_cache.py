from chatchat.configs import CACHED_VS_NUM, CACHED_MEMO_VS_NUM
from chatchat.server.knowledge_base.kb_cache.base import *
from chatchat.server.utils import get_Embeddings
from chatchat.server.knowledge_base.utils import get_vs_path
from chatchat.server.knowledge_base.kb_cache.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import os
from langchain.schema import Document

# patch FAISS to include doc id in Document.metadata
def _new_ds_search(self, search: str) -> Union[str, Document]:
    if search not in self._dict:
        return f"ID {search} not found."
    else:
        doc = self._dict[search]
        if isinstance(doc, Document):
            doc.metadata["id"] = search
        return doc
InMemoryDocstore.search = _new_ds_search


class ThreadSafeFaiss(ThreadSafeObject):
    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"

    def docs_count(self) -> int:
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
         with self.acquire():
             if not os.path.isdir(path) and create_path:
                 os.makedirs(path)
             ret = self._obj.save_local(path)
             logger.info(f"Vector library {self.key} has been saved to disk")
         return ret

    def clear(self):
        ret = []
        with self.acquire():
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"The vector library {self.key} has been cleared")
        return ret


class _FaissPool(CachePool):
    def new_vector_store(
            self,
            kb_name: str,
            embed_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> FAISS:

        # create an empty vector store
        embeddings = get_Embeddings(embed_model=embed_model)
        doc = Document(page_content="init", metadata={})
        # FAISS.__from(normalize_L2 = True, distance_strategy = "COSINE")
        vector_store = FAISS.from_documents([doc], embeddings)
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def new_temp_vector_store(
            self,
            embed_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> FAISS:

        # create an empty vector store
        embeddings = get_Embeddings(embed_model=embed_model)
        doc = Document(page_content="init", metadata={})
        vector_store = FAISS.from_documents([doc], embeddings)
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def save_vector_store(self, kb_name: str, path: str = None):
        if cache := self.get(kb_name):
            return cache.save(path)

    def unload_vector_store(self, kb_name: str):
        if cache := self.get(kb_name):
            self.pop(kb_name)
            logger.info(f"Succeed to unload vector store {kb_name}")


class KBFaissPool(_FaissPool):
    def load_vector_store(
            self,
            kb_name: str,
            vector_name: str = None,
            create: bool = True,
            embed_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> ThreadSafeFaiss:
        self.atomic.acquire()
        vector_name = vector_name or embed_model
        cache = self.get((kb_name, vector_name)) 
        try:
            if cache is None:
                item = ThreadSafeFaiss((kb_name, vector_name), pool=self)
                self.set((kb_name, vector_name), item)
                with item.acquire(msg="Initializing"):
                    logger.info(f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk.")
                    vs_path = get_vs_path(kb_name, vector_name)

                    if os.path.isfile(os.path.join(vs_path, "index.faiss")):
                        embeddings = get_Embeddings(embed_model=embed_model)
                        vector_store = FAISS.load_local(vs_path, embeddings, distance_strategy="COSINE", allow_dangerous_deserialization=True)
                        logger.info(f"Vector library {kb_name} has been loaded from disk.")
                    elif create:
                        # create an empty vector store
                        if not os.path.exists(vs_path):
                            os.makedirs(vs_path)
                        vector_store = self.new_vector_store(kb_name=kb_name, embed_model=embed_model)
                        logger.info(f"Vector library {kb_name} has been created.")
                        vector_store.save_local(vs_path)
                    else:
                        raise RuntimeError(f"knowledge base {kb_name} not exist.")
                    item.obj = vector_store
                    item.finish_loading()
            self.atomic.release()
        except Exception as e:
            self.atomic.release()
            logger.error(e)
            raise RuntimeError(f"Vector library {kb_name} failed to load.")
        return self.get((kb_name, vector_name))


class MemoFaissPool(_FaissPool):
    r"""
    临时向量库的缓存池
    """
    def load_vector_store(
            self,
            kb_name: str,
            embed_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> ThreadSafeFaiss:
        self.atomic.acquire()
        cache = self.get(kb_name)
        if cache is None:
            item = ThreadSafeFaiss(kb_name, pool=self)
            self.set(kb_name, item)
            with item.acquire(msg="Initializing"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}' to memory.")
                # create an empty vector store
                vector_store = self.new_temp_vector_store(embed_model=embed_model)
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(kb_name)


kb_faiss_pool = KBFaissPool(cache_num=CACHED_VS_NUM)
memo_faiss_pool = MemoFaissPool(cache_num=CACHED_MEMO_VS_NUM)