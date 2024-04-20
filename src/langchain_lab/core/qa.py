# Copyright 2023 Lei Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models.base import BaseChatModel
from langchain.docstore.document import Document

from src.langchain_lab.core.embedding import FolderIndex
from src.langchain_lab.core.llm import TrackerCallbackHandler, TrackItem


class AnswerWithSources:
    answer: str
    sources: List[Document]
    relevant_docs: List[Document]
    tracks: List[TrackItem]

    def __init__(
        self,
        answer: str,
        sources: List[Document],
        relevant_docs: List[Document],
        tracks: List[TrackItem],
    ):
        self.answer = answer
        self.sources = sources
        self.relevant_docs = relevant_docs
        self.tracks = tracks


def query_folder(
    query: str,
    llm: BaseChatModel,
    folder_index: FolderIndex,
    top_k: int = 5,
    summary_language: str = "English",
    chain_type: str = "stuff",
    callback: TrackerCallbackHandler = None,
) -> AnswerWithSources:
    if chain_type == "stuff":
        chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type=chain_type,
            #prompt=STUFF_PROMPT,
            callbacks=[callback],
        )
    elif chain_type == "refine":
        chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type=chain_type,
            callbacks=[callback],
        )
    elif chain_type == "map_reduce":
        chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type=chain_type,
            callbacks=[callback],
        )
    elif chain_type == "map_rerank":
        chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type=chain_type,
            callbacks=[callback],
        )
    relevant_docs = folder_index.index.similarity_search(query, k=top_k)

    try:
        # summary_language = detect_language(selection=query)
        result = chain(
            {
                "input_documents": relevant_docs,
                "question": query,
                "lang": summary_language,
            },
            return_only_outputs=True,
        )
        sources = get_sources(result["output_text"], folder_index)
        answer = result["output_text"].split("SOURCES: ")[0]

        # Translate answer to summary language
        # answer = translate(selection=answer, language=summary_language, llm=llm, callback=callback)
    except Exception as e:
        answer = str(e)
        sources = []
    tracks = callback.get_tracks()
    try:
        return AnswerWithSources(answer=answer, sources=sources, relevant_docs=relevant_docs, tracks=tracks)
    finally:
        callback.clean_tracks()


def get_sources(answer: str, folder_index: FolderIndex) -> List[Document]:
    """Retrieves the docs that were used to answer the question the generated answer."""

    source_keys = [s for s in answer.split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in folder_index.docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs
