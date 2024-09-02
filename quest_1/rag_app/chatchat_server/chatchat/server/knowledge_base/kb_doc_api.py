import os
import urllib
from fastapi import File, Form, Body, Query, UploadFile
from chatchat.configs import (DEFAULT_VS_TYPE, DEFAULT_EMBEDDING_MODEL,
                     VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE,
                     logger, log_verbose, )
from chatchat.server.utils import BaseResponse, ListResponse, run_in_thread_pool
from chatchat.server.knowledge_base.utils import (validate_kb_name, list_files_from_folder, get_file_path,
                                         files2docs_in_thread, KnowledgeFile)
from fastapi.responses import FileResponse
from sse_starlette import EventSourceResponse
import json
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.db.repository.knowledge_file_repository import get_file_detail
from langchain.docstore.document import Document
from chatchat.server.knowledge_base.model.kb_document_model import DocumentWithVSId
from typing import List, Dict


def search_docs(
        query: str = Body("", description="User input", examples=["Hello"]),
        knowledge_base_name: str = Body(..., description="knowledge base name", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="Number of matching vectors"),
        score_threshold: float = Body(SCORE_THRESHOLD,
                                    description="Knowledge base matching relevance threshold, value range is between 0-1,"
                                                "The smaller the SCORE, the higher the correlation,"
                                                "A value of 1 is equivalent to no filtering, and it is recommended to set it to around 0.5",
                                    ge=0, le=1),
        file_name: str = Body("", description="File name, supports sql wildcard"),
        metadata: dict = Body({}, description="Filtering based on metadata, only supports first-level keys"),
) -> List[Dict]:
    logger.info(f'knowledge base name:{knowledge_base_name}')
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    if kb is not None:
        if query:
            logger.info(f'knowledge base query:{query}')
            docs = kb.search_docs(query, top_k, score_threshold)
            # data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
            logger.info('knowledge base docs')
            data = [DocumentWithVSId(**x.dict(), id=x.metadata.get("id")) for x in docs]
        elif file_name or metadata:
            logger.info(f'knowledge base file name:{file_name}')
            data = kb.list_docs(file_name=file_name, metadata=metadata)
            for d in data:
                if "vector" in d.metadata:
                    del d.metadata["vector"]
    logger.info(f'knowledge base data size: {len(data)}')
    return [x.dict() for x in data]


def list_files(
        knowledge_base_name: str
) -> ListResponse:
    if not validate_kb_name(knowledge_base_name):
        return ListResponse(code=403, msg="Don't attack me", data=[])

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return ListResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
    else:
        all_doc_names = kb.list_files()
        return ListResponse(data=all_doc_names)


def _save_files_in_thread(files: List[UploadFile],
                           knowledge_base_name: str,
                           override: bool):
    """
    Save the uploaded files to the corresponding knowledge base directory through multi-threading.
    The generator returns the saved result: {"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    """

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        '''
        Save a single file.
        '''
        try:
            filename = file.filename
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read() # Read the content of the uploaded file
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                file_status = f"File {filename} already exists."
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"Successfully uploaded file {filename}", data=data)
        except Exception as e:
            msg = f"{filename} file upload failed, the error message is: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                        exc_info=e if log_verbose else None)
            return dict(code=500, msg=msg, data=data)

    params = [{"file": file, "knowledge_base_name": knowledge_base_name, "override": override} for file in files]
    for result in run_in_thread_pool(save_file, params=params):
        yield result


# def files2docs(files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
#                 knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
#                 override: bool = Form(False, description="覆盖已有文件"),
#                 save: bool = Form(True, description="是否将文件保存到知识库目录")):
#     def save_files(files, knowledge_base_name, override):
#         for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
#             yield json.dumps(result, ensure_ascii=False)

#     def files_to_docs(files):
#         for result in files2docs_in_thread(files):
#             yield json.dumps(result, ensure_ascii=False)


def upload_docs(
        files: List[UploadFile] = File(..., description="Upload files, supports multiple files"),
        knowledge_base_name: str = Form(..., description="knowledge base name", examples=["samples"]),
        override: bool = Form(False, description="Override existing file"),
        to_vector_store: bool = Form(True, description="Whether to vectorize the file after uploading it"),
        chunk_size: int = Form(CHUNK_SIZE, description="Maximum length of a single paragraph of text in the knowledge base"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="The overlap length of adjacent texts in the knowledge base"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="Whether to enable Chinese title enhancement"),
        docs: str = Form("", description="Customized docs, need to be converted to json string"),
        not_refresh_vs_cache: bool = Form(False, description="Do not save vector library (for FAISS)"),
) -> BaseResponse:
    """
    API interface: upload files and/or vectorize
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Knowledge base {knowledge_base_name} not found")

    docs = json.loads(docs) if docs else {}
    failed_files = {}
    file_names = list(docs.keys())

    # First save the uploaded file to disk
    for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    # Vectorize the saved file
    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return BaseResponse(code=200, msg="File upload and vectorization completed", data={"failed_files": failed_files})


def delete_docs(
         knowledge_base_name: str = Body(..., examples=["samples"]),
         file_names: List[str] = Body(..., examples=[["file_name.md", "test.txt"]]),
         delete_content: bool = Body(False),
         not_refresh_vs_cache: bool = Body(False, description="Do not save vector library (for FAISS)"),
) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Knowledge base {knowledge_base_name} not found")

    failed_files = {}
    for file_name in file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"File {file_name} not found"

        try:
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"{file_name} file deletion failed, error message: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                        exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"File deletion completed", data={"failed_files": failed_files})

def update_info(
         knowledge_base_name: str = Body(..., description="knowledge base name", examples=["samples"]),
         kb_info: str = Body(..., description="Introduction to the knowledge base", examples=["This is a knowledge base"]),
):
     if not validate_kb_name(knowledge_base_name):
         return BaseResponse(code=403, msg="Don't attack me")

     kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
     if kb is None:
         return BaseResponse(code=404, msg=f"Knowledge base {knowledge_base_name} not found")
     kb.update_info(kb_info)

     return BaseResponse(code=200, msg=f"Knowledge base introduction modification completed", data={"kb_info": kb_info})


def update_docs(
        knowledge_base_name: str = Body(..., description="knowledge base name", examples=["samples"]),
        file_names: List[str] = Body(..., description="File name, supports multiple files", examples=[["file_name1", "text.txt"]]),
        chunk_size: int = Body(CHUNK_SIZE, description="Maximum length of a single paragraph of text in the knowledge base"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="The overlap length of adjacent texts in the knowledge base"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="Whether to enable Chinese title enhancement"),
        override_custom_docs: bool = Body(False, description="Whether to override previously customized docs"),
        docs: str = Body("", description="Customized docs, need to be converted to json string"),
        not_refresh_vs_cache: bool = Body(False, description="Do not save vector library (for FAISS)"),
) -> BaseResponse:
    """
    Update knowledge base documentation
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Knowledge base {knowledge_base_name} not found")

    failed_files = {}
    kb_files = []
    docs = json.loads(docs) if docs else {}

    # Generate a list of files that need to load docs
    for file_name in file_names:
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # If the file has previously used custom docs, it will be skipped or overwritten based on the parameters.
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs:
            try:
                kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name))
            except Exception as e:
                msg = f"Error loading document {file_name}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                            exc_info=e if log_verbose else None)
                failed_files[file_name] = msg

    # Generate docs from files and vectorize.
    # This uses the caching function of KnowledgeFile to load the Document in multiple threads and then passes it to KnowledgeFile.
    for status, result in files2docs_in_thread(kb_files,
                                            chunk_size=chunk_size,
                                            chunk_overlap=chunk_overlap,
                                            zh_title_enhance=zh_title_enhance):
        if status:
            kb_name, file_name, new_docs = result
  
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb_file.splited_docs = new_docs

            kb.update_doc(kb_file, not_refresh_vs_cache=True)
            logger.info(f"Update document {result[1]} completed")
        else:
            logger.error(f"Error adding file '{result[1]}' to knowledge base '{result[0]}': {result[2]}. Skipped.")
            kb_name, file_name, error = result
            failed_files[file_name] = error

    # Vectorize custom docs
    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
            kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"Error adding custom docs for {file_name}: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                        exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"Update document completed", data={"failed_files": failed_files})


def download_doc(
        knowledge_base_name: str = Query(..., description="knowledge base name", examples=["samples"]),
        file_name: str = Query(..., description="file name", examples=["test.txt"]),
        preview: bool = Query(False, description="Yes: Preview in browser; No: Download"),
):
    """
    Download knowledge base documents
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Knowledge base {knowledge_base_name} not found")

    if preview:
        content_disposition_type = "inline"
    else:
        content_disposition_type = None

    try:
        kb_file = KnowledgeFile(filename=file_name,
                                knowledge_base_name=knowledge_base_name)

        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"{kb_file.filename} failed to read the file, the error message is: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                    exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"{kb_file.filename} failed to read file")


def recreate_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(DEFAULT_EMBEDDING_MODEL),
        chunk_size: int = Body(CHUNK_SIZE, description="Maximum length of a single paragraph of text in the knowledge base"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="The overlap length of adjacent texts in the knowledge base"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="Whether to enable Chinese title enhancement"),
        not_refresh_vs_cache: bool = Body(False, description="Do not save vector library (for FAISS)"),
):
    """
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
    set allow_empty_kb to True make it applied on empty knowledge base which it not in the info.db or having no documents.
    """

    def output():
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"Knowledge base {knowledge_base_name} not found"}
        else:
            if kb.exists():
                kb.clear_vs()
            kb.create_kb()
            files = list_files_from_folder(knowledge_base_name)
            kb_files = [(file, knowledge_base_name) for file in files]
            i = 0
            for status, result in files2docs_in_thread(kb_files,
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    zh_title_enhance=zh_title_enhance):
                if status:
                    kb_name, file_name, docs = result
                    kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
                    kb_file.splited_docs = docs
                    yield json.dumps({
                        "code": 200,
                        "msg": f"({i + 1} / {len(files)}): {file_name}",
                        "total": len(files),
                        "finished": i + 1,
                        "doc": file_name,
                    }, ensure_ascii=False)
                    kb.add_doc(kb_file, not_refresh_vs_cache=True)
                else:
                    kb_name, file_name, error = result
                    msg = f"Error adding file '{file_name}' to knowledge base '{knowledge_base_name}': {error}. Skipped."
                    logger.error(msg)
                    yield json.dumps({
                        "code": 500,
                        "msg": msg,
                    })
                i += 1
            if not not_refresh_vs_cache:
                kb.save_vector_store()

    return EventSourceResponse(output())