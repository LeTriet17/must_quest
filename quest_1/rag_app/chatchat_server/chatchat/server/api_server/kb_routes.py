from __future__ import annotations

from typing import List

from fastapi import APIRouter, Request

from chatchat.server.chat.file_chat import upload_temp_docs
from chatchat.server.knowledge_base.kb_api import list_kbs, create_kb, delete_kb
from chatchat.server.knowledge_base.kb_doc_api import (list_files, upload_docs, delete_docs,
                                            update_docs, download_doc, recreate_vector_store,
                                            search_docs, update_info)

from chatchat.server.utils import BaseResponse, ListResponse


kb_router = APIRouter(prefix="/knowledge_base", tags=["Knowledge Base Management"])


kb_router.get("/list_knowledge_bases",
        response_model=ListResponse,
        summary="List all knowledge bases"
        )(list_kbs)

kb_router.post("/create_knowledge_base",
            response_model=BaseResponse,
            summary="Create a new knowledge base"
            )(create_kb)

kb_router.post("/delete_knowledge_base",
            response_model=BaseResponse,
            summary="Delete a knowledge base"
            )(delete_kb)

kb_router.get("/list_files",
        response_model=ListResponse,
        summary="List all files in a knowledge base"
        )(list_files)

kb_router.post("/search_docs",
            response_model=List[dict],
            summary="Search documents in a knowledge base"
            )(search_docs)

kb_router.post("/upload_docs",
            response_model=BaseResponse,
            summary="Upload files to a knowledge base"
            )(upload_docs)

kb_router.post("/delete_docs",
            response_model=BaseResponse,
            summary="Delete files from a knowledge base"
            )(delete_docs)

kb_router.post("/update_info",
            response_model=BaseResponse,
            summary="Update file info"
            )(update_info)

kb_router.post("/update_docs",
            response_model=BaseResponse,
            summary="Update files in a knowledge base"
            )(update_docs)

kb_router.get("/download_doc",
            summary="Download a file"
            )(download_doc)

kb_router.post("/recreate_vector_store",
            summary="Recreate vector store"
            )(recreate_vector_store)

kb_router.post("/upload_temp_docs",
            summary="Upload temporary files to a knowledge base"
            )(upload_temp_docs)
