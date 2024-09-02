from abc import ABC, abstractmethod

import operator
import os
from pathlib import Path
from langchain.docstore.document import Document

from chatchat.server.db.repository.knowledge_base_repository import (
    add_kb_to_db, delete_kb_from_db, list_kbs_from_db, kb_exists,
    load_kb_from_db, get_kb_detail,
)
from chatchat.server.db.repository.knowledge_file_repository import (
    add_file_to_db, delete_file_from_db, delete_files_from_db, file_exists_in_db,
    count_files_from_db, list_files_from_db, get_file_detail, delete_file_from_db,
    list_docs_from_db,
)

import asyncio
from chatchat.configs import (kbs_config, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     DEFAULT_EMBEDDING_MODEL, KB_INFO)
from chatchat.server.knowledge_base.utils import (
    get_kb_path, get_doc_path, get_table_path, KnowledgeFile,
    list_kbs_from_folder, list_files_from_folder
)

from typing import List, Union, Dict, Optional, Tuple

from chatchat.server.knowledge_base.model.kb_document_model import DocumentWithVSId


class SupportedVSType:
    FAISS = 'faiss'
    DEFAULT = 'default'
    PG = 'pg'
    CHROMADB = 'chromadb'


class KBService(ABC):

    def __init__(self,
                 knowledge_base_name: str,
                 embed_model: str = DEFAULT_EMBEDDING_MODEL,
                 ):
        self.kb_name = knowledge_base_name
        self.kb_info = KB_INFO.get(knowledge_base_name, f"{knowledge_base_name} knowledge base")
        self.embed_model = embed_model
        self.kb_path = get_kb_path(self.kb_name)
        self.doc_path = get_doc_path(self.kb_name)
        self.table_path = get_table_path(self.kb_name)
        self.do_init()

    def __repr__(self) -> str:
        return f"{self.kb_name} @ {self.embed_model}"

    def save_vector_store(self):
        '''
        Save the vector store: FAISS saves to disk, milvus saves to the database. PGVector is not yet supported.
        '''
        pass

    def create_kb(self):
        """
        Create a knowledge base
        """
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)
        if not os.path.exists(self.table_path):
            os.makedirs(self.table_path)
        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)

        if status:
            self.do_create_kb()
        return status

    def clear_vs(self):
        """
        Delete all content in the vector store
        """
        self.do_clear_vs()
        status = delete_files_from_db(self.kb_name)
        return status

    def drop_kb(self):
        """
        Delete the knowledge base
        """
        self.do_drop_kb()
        status = delete_kb_from_db(self.kb_name)
        return status

    def _docs_to_embeddings(self, docs: List[Document]) -> Dict:
        '''
        将 List[Document] 转化为 VectorStore.add_embeddings 可以接受的参数
        '''
        return embed_documents(docs=docs, embed_model=self.embed_model, to_query=False)
    
    def add_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        Add files to the knowledge base
        If docs are specified, no longer vectorize the text and mark the corresponding database entry as custom_docs=True
        """

        def get_event_loop():
            try:
                return asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop

        if docs:
            custom_docs = True
        else:
            loop = get_event_loop()
            docs =  loop.run_until_complete((kb_file.file2text()))
            custom_docs = False

        if docs:
            # Change metadata["source"] to relative path
            for doc in docs:
                try:
                    doc.metadata.setdefault("source", kb_file.filename)
                    source = doc.metadata.get("source", "")
                    if os.path.isabs(source):
                        rel_path = Path(source).relative_to(self.doc_path)
                        doc.metadata["source"] = str(rel_path.as_posix().strip("/"))
                except Exception as e:
                    print(f"cannot convert absolute path ({source}) to relative path. error is : {e}")
            self.delete_doc(kb_file)
            doc_infos = self.do_add_doc(docs, **kwargs)
            status = add_file_to_db(kb_file,
                                    custom_docs=custom_docs,
                                    docs_count=len(docs),
                                    doc_infos=doc_infos)
        else:
            status = False
        return status

    def delete_doc(self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs):
        """
        Delete files from the knowledge base
        """
        self.do_delete_doc(kb_file, **kwargs)
        status = delete_file_from_db(kb_file)
        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)
        return status

    def update_info(self, kb_info: str):
        """
        Update the knowledge base description
        """
        self.kb_info = kb_info
        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)
        return status

    def update_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        Update the vector store with files from content
        If docs are specified, use custom docs and mark the corresponding database entry as custom_docs=True
        """
        if os.path.exists(kb_file.filepath):
            self.delete_doc(kb_file, **kwargs)
            return self.add_doc(kb_file, docs=docs, **kwargs)

    def exist_doc(self, file_name: str):
        return file_exists_in_db(KnowledgeFile(knowledge_base_name=self.kb_name,
                                               filename=file_name))

    def list_files(self):
        return list_files_from_db(self.kb_name)

    def count_files(self):
        return count_files_from_db(self.kb_name)

    def search_docs(self,
                    query: str,
                    top_k: int = VECTOR_SEARCH_TOP_K,
                    score_threshold: float = SCORE_THRESHOLD,
                    ) ->List[Document]:
        docs = self.do_search(query, top_k, score_threshold)
        for doc in docs:
            if doc.metadata.get("table"):
                doc.page_content = doc.metadata["table"]
        return docs

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        return []

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        raise NotImplementedError

    def update_doc_by_ids(self, docs: Dict[str, Document]) -> bool:
        '''
        The parameter passed is: {doc_id: Document, ...}
        If the value corresponding to doc_id is None, or its page_content is empty, delete the document
        '''
        self.del_doc_by_ids(list(docs.keys()))
        docs = []
        ids = []
        for k, v in docs.items():
            if not v or not v.page_content.strip():
                continue
            ids.append(k)
            docs.append(v)
        self.do_add_doc(docs=docs, ids=ids)
        return True

    def list_docs(self, file_name: str = None, metadata: Dict = {}) -> List[DocumentWithVSId]:
        '''
        List all documents in the knowledge base
        '''
        doc_infos = list_docs_from_db(kb_name=self.kb_name, file_name=file_name, metadata=metadata)
        docs = []
        for x in doc_infos:
            doc_info = self.get_doc_by_ids([x["id"]])[0]
            if doc_info is not None:
                # Handle non-empty cases
                doc_with_id = DocumentWithVSId(**doc_info.dict(), id=x["id"])
                docs.append(doc_with_id)
            else:
                # Handle empty cases
                # You can choose to skip the current iteration or perform other actions
                pass
        return docs

    @abstractmethod
    def do_create_kb(self):
        """
        Implement logic for creating knowledge base in subclass
        """
        pass

    @staticmethod
    def list_kbs_type():
        return list(kbs_config.keys())

    @classmethod
    def list_kbs(cls):
        return list_kbs_from_db()

    def exists(self, kb_name: str = None):
        kb_name = kb_name or self.kb_name
        return kb_exists(kb_name)

    @abstractmethod
    def vs_type(self) -> str:
        pass

    @abstractmethod
    def do_init(self):
        pass

    @abstractmethod
    def do_drop_kb(self):
        """
        Implement logic for deleting knowledge base in subclass
        """
        pass

    @abstractmethod
    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float,
                  ) -> List[Tuple[Document, float]]:
        """
        Implement logic for searching knowledge base in subclass
        """
        pass

    @abstractmethod
    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:
        """
        Implement logic for adding documents to knowledge base in subclass
        """
        pass

    @abstractmethod
    def do_delete_doc(self,
                      kb_file: KnowledgeFile):
        """
        Implement logic for deleting documents from knowledge base in subclass
        """
        pass

    @abstractmethod
    def do_clear_vs(self):
        """
        Implement logic for clearing all vectors from knowledge base in subclass
        """
        pass

class KBServiceFactory:

    @staticmethod
    def get_service(kb_name: str,
                    vector_store_type: Union[str, SupportedVSType],
                    embed_model: str = DEFAULT_EMBEDDING_MODEL,
                    ) -> KBService:
        if isinstance(vector_store_type, str):
            vector_store_type = getattr(SupportedVSType, vector_store_type.upper())
        if SupportedVSType.FAISS == vector_store_type:
            from chatchat.server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
            return FaissKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.PG == vector_store_type:
            from chatchat.server.knowledge_base.kb_service.pg_kb_service import PGKBService
            return PGKBService(kb_name, embed_model=embed_model)
        if SupportedVSType.FAISS == vector_store_type:
            from chatchat.server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
            return FaissKBService(kb_name, embed_model=embed_model)
        
        elif SupportedVSType.CHROMADB == vector_store_type:
            from chatchat.server.knowledge_base.kb_service.chromadb_kb_service import ChromaKBService
            return ChromaKBService(kb_name, embed_model=embed_model)

    @staticmethod
    def get_service_by_name(kb_name: str) -> KBService:
        _, vs_type, embed_model = load_kb_from_db(kb_name)
        if _ is None:  # kb not in db, just return None
            return None
        return KBServiceFactory.get_service(kb_name, vs_type, embed_model)

    @staticmethod
    def get_default():
        return KBServiceFactory.get_service("default", SupportedVSType.DEFAULT)


def get_kb_details() -> List[Dict]:
    kbs_in_folder = list_kbs_from_folder()
    kbs_in_db = KBService.list_kbs()
    result = {}

    for kb in kbs_in_folder:
        result[kb] = {
            "kb_name": kb,
            "vs_type": "",
            "kb_info": "",
            "embed_model": "",
            "file_count": 0,
            "create_time": None,
            "in_folder": True,
            "in_db": False,
        }

    for kb in kbs_in_db:
        kb_detail = get_kb_detail(kb)
        if kb_detail:
            kb_detail["in_db"] = True
            if kb in result:
                result[kb].update(kb_detail)
            else:
                kb_detail["in_folder"] = False
                result[kb] = kb_detail

    data = []
    for i, v in enumerate(result.values()):
        v['No'] = i + 1
        data.append(v)

    return data


def get_kb_file_details(kb_name: str) -> List[Dict]:
    kb = KBServiceFactory.get_service_by_name(kb_name)
    if kb is None:
        return []

    files_in_folder = list_files_from_folder(kb_name)
    files_in_db = kb.list_files()
    result = {}

    for doc in files_in_folder:
        result[doc] = {
            "kb_name": kb_name,
            "file_name": doc,
            "file_ext": os.path.splitext(doc)[-1],
            "file_version": 0,
            "document_loader": "",
            "docs_count": 0,
            "text_splitter": "",
            "create_time": None,
            "in_folder": True,
            "in_db": False,
        }
    lower_names = {x.lower(): x for x in result}
    for doc in files_in_db:
        doc_detail = get_file_detail(kb_name, doc)
        if doc_detail:
            doc_detail["in_db"] = True
            if doc.lower() in lower_names:
                result[lower_names[doc.lower()]].update(doc_detail)
            else:
                doc_detail["in_folder"] = False
                result[doc] = doc_detail

    data = []
    for i, v in enumerate(result.values()):
        v['No'] = i + 1
        data.append(v)

    return data


def score_threshold_process(score_threshold, k, docs):
    if score_threshold is not None:
        cmp = (
            operator.le
        )
        docs = [
            (doc, similarity)
            for doc, similarity in docs
            if cmp(similarity, score_threshold)
        ]
    return docs[:k]
