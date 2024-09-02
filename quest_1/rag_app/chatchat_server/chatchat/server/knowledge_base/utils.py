import os
from functools import lru_cache
from chatchat.configs import (
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    logger,
    log_verbose,
    TEXT_SPLITTER_NAME,
    DEFAULT_EMBEDDING_MODEL
)
import importlib
from chatchat.server.file_rag.text_splitter import table_aware_markdown_splitter
import langchain_community.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from pathlib import Path
from chatchat.server.utils import run_in_thread_pool, run_in_process_pool
import json
from typing import List, Union, Dict, Tuple, Generator
import chardet
from langchain_community.document_loaders import JSONLoader, TextLoader
import pandas as pd
from io import StringIO
import asyncio
from retrying import retry 

def validate_kb_name(knowledge_base_id: str) -> bool:
    # Validate knowledge base name
    if "../" in knowledge_base_id:
        return False
    return True

def save_to_excel(data_dict, output_file):
    logger.info(f"Saving data to Excel file: {output_file}")
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    for page_num, page in enumerate(data_dict):
        csv_list = page[str(page_num + 1)]["table"]
        for idx, csv_data in enumerate(csv_list):
            # Create a new sheet for each CSV data with sheet_name as "Page_{page_num}_Table_{idx}"
            sheet_name = f"Page_{page_num+1}_Table_{idx+1}"
            # Convert CSV data into DataFrame
            df = pd.read_csv(StringIO(csv_data), sep=",", on_bad_lines='skip')
            # Write DataFrame to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=True, header=True)
    writer.close()
    logger.info(f"Data has been saved to Excel file: {output_file}")

def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)

def get_table_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "table")

def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str, vector_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store", vector_name)

def get_table_doc_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_table_path(knowledge_base_name), doc_name)

def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def list_kbs_from_folder():
    return [f for f in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, f))]


def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    result = []

    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False

    def process_entry(entry):
        if is_skiped_path(entry.path):
            return

        if entry.is_symlink():
            target_path = os.path.realpath(entry.path)
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    process_entry(target_entry)
        elif entry.is_file():
            file_path = (Path(os.path.relpath(entry.path, doc_path)).as_posix()) # 路径统一为 posix 格式
            result.append(file_path)
        elif entry.is_dir():
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    process_entry(sub_entry)

    with os.scandir(doc_path) as it:
        for entry in it:
            process_entry(entry)

    return result


LOADER_DICT = {"LLamaParseLoader": ['.pdf', '.docx', '.doc', '.ppt', '.pptx', '.xls', '.xlsx'],
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


# patch json.dumps to disable ensure_ascii
def _new_json_dumps(obj, **kwargs):
    kwargs["ensure_ascii"] = False
    return _origin_json_dumps(obj, **kwargs)


if json.dumps is not _new_json_dumps:
    _origin_json_dumps = json.dumps
    json.dumps = _new_json_dumps


class JSONLinesLoader(JSONLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True


langchain_community.document_loaders.JSONLinesLoader = JSONLinesLoader


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            print(f'LoaderClass: {LoaderClass}')
            return LoaderClass


def get_loader(loader_name: str, file_path: str, loader_kwargs: Dict = None):
    '''
    Returns the document loader based on loader_name and file path or content.
    '''
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in ["FilteredCSVLoader", "LLamaParseLoader"]:
            document_loaders_module = importlib.import_module("chatchat.server.file_rag.document_loaders")
        else:
            document_loaders_module = importlib.import_module("langchain_community.document_loaders")
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        msg = f"Error finding loader {loader_name} for file {file_path}: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        document_loaders_module = importlib.import_module("langchain_community.document_loaders")
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, 'rb') as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]

    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader


class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            knowledge_base_name: str,
            loader_kwargs: Dict = {},
    ):
        '''
        The files in the corresponding knowledge base directory must exist on the disk to perform operations such as vectorization.
        '''
        self.kb_name = knowledge_base_name
        self.filename = str(Path(filename).as_posix())
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file format {self.filename}")
        self.loader_kwargs = loader_kwargs
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.tablepath = get_table_doc_path(knowledge_base_name, os.path.splitext(filename)[0] + ".xlsx")
        self.docs = None
        self.splited_docs = None
        self.document_loader_name = get_LoaderClass(self.ext)
        self.text_splitter_name = TEXT_SPLITTER_NAME

    def file2docs(self, refresh: bool = False):
        if self.docs is None or refresh:
            logger.info(f"{self.document_loader_name} used for {self.filepath}")

            def retry_if_exception(exception):
                return isinstance(exception, Exception)
            
            @retry(wait_fixed=1000, stop_max_attempt_number=3, retry_on_exception=retry_if_exception)
            def load_docs_with_retry():
                logger.info(f"Loading documents from {self.filepath}")
                loader = get_loader(loader_name=self.document_loader_name,
                                    file_path=self.filepath,
                                    loader_kwargs=self.loader_kwargs)
                if isinstance(loader, TextLoader):
                    loader.encoding = "utf8"
                try:
                    docs = loader.load()
                except Exception as e:
                    logger.error(f"Failed to load documents: {str(e)}")
                    raise Exception(f"Failed to load documents: {str(e)}")
                if docs is None or len(docs) == 0:
                    logger.error(f"No documents loaded from {self.filepath}, retrying")
                    raise Exception(f"No documents loaded from {self.filepath}")
                return docs

            self.docs = load_docs_with_retry()

        return self.docs

    def docs2texts(
            self,
            docs: List[Document] = None,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    ):
        docs = docs or self.file2docs(refresh=refresh)
        logger.info(f"Texts from {self.filepath} have been loaded")
        assert len(docs) == 2
        tables_data = json.loads(docs[1].page_content)
        save_to_excel(tables_data, self.tablepath)
        docs = docs[0]
        if not docs:
            return []
        if self.ext not in [".csv"]:
            self.splited_docs = table_aware_markdown_splitter(docs, chunk_size=chunk_size)

        logger.info(f"Texts from {self.filepath} have been split")
        if not self.splited_docs:
            return []
        

        return self.splited_docs

    async def file2text(
            self,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    ):
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(docs=docs,
                                                zh_title_enhance=zh_title_enhance,
                                                refresh=refresh,
                                                chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                text_splitter=text_splitter)
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)

def files2docs_in_thread_file2docs(*, file: KnowledgeFile, **kwargs) -> Tuple[bool, Tuple[str, str, List[str]]]:
    
    def get_event_loop():
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    loop = get_event_loop()
    try:
        result = loop.run_until_complete(file.file2text(**kwargs))
        return True, (file.kb_name, file.filename, result)
    except Exception as e:
        msg = f"Error loading document from file {file.kb_name}/{file.filename}: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e if log_verbose else None)
        return False, (file.kb_name, file.filename, msg)

def files2docs_in_thread(
        files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
) -> Generator:
    '''
        Use multi-threading to batch convert disk files into langchain Document.
        If the incoming parameter is a Tuple, the form is (filename, kb_name)
        The generator return value is status, (kb_name, file_name, docs | error)
    '''

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    for result in run_in_thread_pool(func=files2docs_in_thread_file2docs, params=kwargs_list):
        print('Calling files2docs_in_thread_file2docs')
        yield result


if __name__ == "__main__":
    from pprint import pprint

    kb_file = KnowledgeFile(
        filename="E:\\LLM\\Data\\Test.md",
        knowledge_base_name="samples")
    # kb_file.text_splitter_name = "RecursiveCharacterTextSplitter"
    kb_file.text_splitter_name = "MarkdownHeaderTextSplitter"
    docs = kb_file.file2docs()
    # pprint(docs[-1])
    texts = kb_file.docs2texts(docs)
    for text in texts:
        print(text)