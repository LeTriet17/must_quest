from typing import List, Optional

from langchain.schema.language_model import BaseLanguageModel

from chatchat.server.knowledge_base.model.kb_document_model import DocumentWithVSId
from chatchat.configs import logger
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate

from langchain.docstore.document import Document
from langchain.output_parsers.regex import RegexParser
from langchain.chains.combine_documents.map_reduce import ReduceDocumentsChain, MapReduceDocumentsChain

import sys
import asyncio


class SummaryAdapter:
    _OVERLAP_SIZE: int
    token_max: int
    _separator: str = "\n\n"
    chain: MapReduceDocumentsChain

    def __init__(self, overlap_size: int, token_max: int,
                 chain: MapReduceDocumentsChain):
        self._OVERLAP_SIZE = overlap_size
        self.chain = chain
        self.token_max = token_max

    @classmethod
    def form_summary(cls,
                     llm: BaseLanguageModel,
                     reduce_llm: BaseLanguageModel,
                     overlap_size: int,
                     token_max: int = 1300):
        """
        Get an instance
        :param reduce_llm: llm used to merge summaries
        :param llm: llm used to generate summaries
        :param overlap_size: size of overlap
        :param token_max: maximum number of chunks, each chunk length less than token_max, summaries greater than token_max length will raise error when generated for the first time
        :return:
        """

        # This controls how each document will be formatted. Specifically,
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )

        # The prompt here should take as an input variable the
        # `document_variable_name`
        prompt_template = (
            "Perform tasks based on text. The following task information "
            "{task_briefing}"
            "Text content is as follows: "
            "\r\n"
            "{context}"
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["task_briefing", "context"]
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # We now define how to combine these summaries
        reduce_prompt = PromptTemplate.from_template(
            "Combine these summaries: {context}"
        )
        reduce_llm_chain = LLMChain(llm=reduce_llm, prompt=reduce_prompt)

        document_variable_name = "context"
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )
        reduce_documents_chain = ReduceDocumentsChain(
            token_max=token_max,
            combine_documents_chain=combine_documents_chain,
        )
        chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name=document_variable_name,
            reduce_documents_chain=reduce_documents_chain,
            # Return intermediate steps
            return_intermediate_steps=True
        )
        return cls(overlap_size=overlap_size,
                   chain=chain,
                   token_max=token_max)

    def summarize(self,
                  file_description: str,
                  docs: List[DocumentWithVSId] = []
                  ) -> List[Document]:

        if sys.version_info < (3, 10):
            loop = asyncio.get_event_loop()
        else:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()

            asyncio.set_event_loop(loop)
        # Synchronously call coroutine code
        return loop.run_until_complete(self.asummarize(file_description=file_description,
                                                       docs=docs))

    async def asummarize(self,
                         file_description: str,
                         docs: List[DocumentWithVSId] = []) -> List[Document]:

        logger.info("Start summarizing")
        """
        This process is divided into two parts:
        1. Process each document to obtain a summary of each document
         map_results = self.llm_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        2. Combine the summaries of each document to obtain the final summary, return_intermediate_steps=True, return intermediate steps
        result, extra_return_dict = self.reduce_documents_chain.combine_docs(
            result_docs, token_max=token_max, callbacks=callbacks, **kwargs
        )
        """
        summary_combine, summary_intermediate_steps = self.chain.combine_docs(docs=docs,
                                                                              task_briefing="Describe the closeness and similarity between different methods to help readers understand their relationship.")
        print(summary_combine)
        print(summary_intermediate_steps)

        # if len(summary_combine) == 0:
        #     # Regenerate if empty, reduce the quantity by half
        #     result_docs = [
        #         Document(page_content=question_result_key, metadata=docs[i].metadata)
        #         # This uses metadata from the docs, and the textual results from `results`
        #         for i, question_result_key in enumerate(
        #             summary_intermediate_steps["intermediate_steps"][
        #             :len(summary_intermediate_steps["intermediate_steps"]) // 2
        #             ])
        #     ]
        #     summary_combine, summary_intermediate_steps = self.chain.reduce_documents_chain.combine_docs(
        #         result_docs, token_max=self.token_max
        #     )
        logger.info("end summary")
        doc_ids = ",".join([doc.id for doc in docs])
        _metadata = {
            "file_description": file_description,
            "summary_intermediate_steps": summary_intermediate_steps,
            "doc_ids": doc_ids
        }
        summary_combine_doc = Document(page_content=summary_combine, metadata=_metadata)

        return [summary_combine_doc]

    def _drop_overlap(self, docs: List[DocumentWithVSId]) -> List[str]:
        """
         # Remove overlapping parts of sentences in the document page_content
        :param docs:
        :param separator:
        :return:
        """
        merge_docs = []

        pre_doc = None
        for doc in docs:
            # Add the first document directly
            if len(merge_docs) == 0:
                pre_doc = doc.page_content
                merge_docs.append(doc.page_content)
                continue

            # Remove overlapping parts of the previous end and the next beginning in the list
            # Iteratively reduce the length of pre_doc, delete the characters in front of it each time,
            # Query the overlapping part until the length of pre_doc is less than self._OVERLAP_SIZE // 2 - 2len(separator)
            for i in range(len(pre_doc), self._OVERLAP_SIZE // 2 - 2 * len(self._separator), -1):
                # Delete characters in front of it each time
                pre_doc = pre_doc[1:]
                if doc.page_content[:len(pre_doc)] == pre_doc:
                    # Delete overlapping parts of the next beginning
                    merge_docs.append(doc.page_content[len(pre_doc):])
                    break

            pre_doc = doc.page_content

        return merge_docs

    def _join_docs(self, docs: List[str]) -> Optional[str]:
        text = self._separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text