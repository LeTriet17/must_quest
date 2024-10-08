import urllib
from chatchat.server.utils import BaseResponse, ListResponse
from chatchat.server.knowledge_base.utils import validate_kb_name
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.db.repository.knowledge_base_repository import list_kbs_from_db
from chatchat.configs import DEFAULT_EMBEDDING_MODEL, logger, log_verbose
from fastapi import Body


def list_kbs():
    # Get List of Knowledge Base
    return ListResponse(data=list_kbs_from_db())


def create_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
              vector_store_type: str = Body("faiss"),
              embed_model: str = Body(DEFAULT_EMBEDDING_MODEL),
              ) -> BaseResponse:
    # Create selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="The knowledge base name cannot be empty")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is not None:
        return BaseResponse(code=404, msg=f"A knowledge base with the name {knowledge_base_name} already exists")

    kb = KBServiceFactory.get_service(knowledge_base_name, vector_store_type, embed_model)
    try:
        kb.create_kb()
    except Exception as e:
        msg = f"Error creating knowledge base: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"A knowledge base with the name {knowledge_base_name} was successfully created")


def delete_kb(
         knowledge_base_name: str = Body(..., examples=["samples"])
) -> BaseResponse:
    # Delete selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)

    if kb is None:
        return BaseResponse(code=404, msg=f"Knowledge base {knowledge_base_name} not found")

    try:
        status = kb.clear_vs()
        status = kb.drop_kb()
        if status:
            return BaseResponse(code=200, msg=f"Knowledge base {knowledge_base_name} was successfully deleted")
    except Exception as e:
        msg = f"An unexpected error occurred while deleting the knowledge base: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                    exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"Failed to delete knowledge base {knowledge_base_name}")