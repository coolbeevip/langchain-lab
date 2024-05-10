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
import re
from datetime import datetime
from typing import Any, List

import streamlit as st
from bs4 import BeautifulSoup as Soup
from langchain_core.documents import Document

from langchain_lab import logger
from langchain_lab.core.summary import summarize
from langchain_lab.langchain_community.document_loaders.recursive_url_loader import (
    RecursiveUrlLoader,
)
from langchain_lab.scenarios.error import display_error
from src.langchain_lab.core.chunking import chunk_file
from src.langchain_lab.core.embedding import embed_docs
from src.langchain_lab.core.parsing import File, read_file
from src.langchain_lab.core.qa import query_folder
from src.langchain_lab.scenarios.debug import show_debug


@st.cache_resource
def splitting_url(url: str, chunk_size: int, chunk_overlap: int):
    start_time = datetime.now()
    with st.spinner(f"Spitting {url}. This may take a while⏳"):
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=2,
            extractor=lambda x: Soup(x, "html.parser").text,
        )
        docs = loader.load()
        final_docs = []
        for doc in docs:
            content_type = doc.metadata.get("content_type", "text/html")
            if content_type.startswith("text/html"):
                page_content = doc.page_content.strip()
                page_content = re.sub(r"\n{2,}", "\n", page_content)
                doc.page_content = page_content
                final_docs.append(doc)

        with st.expander(f"Document {len(final_docs)}"):
            for index, doc in enumerate(final_docs):
                if index > 0:
                    st.write("---")
                st.write(f"📄 {index}-{doc.metadata['source']}")
                try:
                    # Remove extra newlines
                    text = re.sub("\n{2,}", "\n", doc.page_content)
                    st.text(text)
                except Exception as e:
                    st.text(doc.page_content.encode("utf-8", "replace").decode())
                    logger.warning(e)
            end_time = datetime.now()
            time_diff = end_time - start_time
            seconds_diff = time_diff.total_seconds()
        st.info(
            f"Document successfully divided into **{len(final_docs)}** sections, \
               each with a size of **{chunk_size}**, featuring an overlap of **{chunk_overlap}** ({seconds_diff}s)."
        )
    return final_docs


@st.cache_resource
def splitting_file(uploaded_file, chunk_size: int, chunk_overlap: int):
    __file = read_file(uploaded_file)
    if not is_file_valid(__file):
        st.stop()
    start_time = datetime.now()
    with st.spinner(f"Spitting document {uploaded_file.name}. This may take a while⏳"):
        docs = chunk_file(__file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        with st.expander(f"Document {len(docs)}"):
            for index, doc in enumerate(docs):
                if index > 0:
                    st.write("---")
                st.write(f"📄 {index}-{doc.metadata['source']}")
                try:
                    st.text(doc.page_content)
                except Exception as e:
                    st.text(doc.page_content.encode("utf-8", "replace").decode())
                    logger.warning(e)
            end_time = datetime.now()
            time_diff = end_time - start_time
            seconds_diff = time_diff.total_seconds()
        st.info(
            f"Document successfully divided into **{len(docs)}** sections, \
            each with a size of **{chunk_size}**, featuring an overlap of **{chunk_overlap}** ({seconds_diff}s)."
        )
        return docs


@st.cache_resource
def indexing_documents(file_name: str, embedding_model, _docs: List[Document], cache_flag: Any = None):
    try:
        start_time = datetime.now()
        with st.spinner(f"Indexing **{file_name}** This may take a while⏳"):
            folder_index = embed_docs(
                docs=_docs,
                vector_store="faiss",
                embedding=st.session_state["EMBEDDING"],
            )
            st.session_state["folder_index"] = folder_index
            end_time = datetime.now()
            time_diff = end_time - start_time
            seconds_diff = time_diff.total_seconds()
            st.info(f"Completed **{len(_docs)}** sections indexes by **{embedding_model}**! ({seconds_diff}s)")

    except Exception as e:
        logger.error(e)
        display_error(e)


@st.cache_resource
def summarize_documents(_docs: List[Document], cache_flag: Any = None):
    if st.session_state.get("SUMMARIZE", False):
        with st.expander("Summary", True):
            with st.spinner("Wait for summarize...⏳"):
                response = summarize(docs=_docs, llm=st.session_state["LLM"], summary_language=st.session_state["SUMMARY_LANGUAGE"], callback=st.session_state["DEBUG_CALLBACK"])
                st.markdown(response)


def is_file_valid(file: File) -> bool:
    if len(file.docs) == 0 or "".join([doc.page_content for doc in file.docs]).strip() == "":
        st.error("Cannot read document! Make sure the document has selectable text")
        logger.error("Cannot read document")
        return False
    return True


def is_query_valid(query: str) -> bool:
    if not query:
        st.error("Please enter a question!")
        return False
    return True


def init_document_scenario():
    document_type = st.session_state["DOCUMENT_TYPE"]
    logger.info(f"Initializing document[{document_type}] scenario")

    if document_type == "WEB":
        with st.form(key="web_loader_form"):
            url_input = st.text_input("Enter a url", value="https://blog.langchain.dev/", placeholder="https://blog.langchain.dev/")
            load_btn = st.form_submit_button("Load Documents")

            btn_clicked = False
            if load_btn:
                docs = splitting_url(url=url_input, chunk_size=st.session_state["CHUNK_SIZE"], chunk_overlap=st.session_state["CHUNK_OVERLAP"])
                summarize_documents(docs, cache_flag=datetime.now())
                indexing_documents(
                    file_name=url_input,
                    embedding_model=st.session_state["EMBED_MODEL_NAME"],
                    cache_flag=datetime.now(),
                    _docs=docs,
                )
                btn_clicked = True

            if "folder_index" not in st.session_state:
                st.stop()
            else:
                if not btn_clicked:
                    splitting_url(url=url_input, chunk_size=st.session_state["CHUNK_SIZE"], chunk_overlap=st.session_state["CHUNK_OVERLAP"])

    elif document_type == "FILE":
        file = st.file_uploader(
            "Upload a pdf, docx, txt or csv file",
            type=["pdf", "docx", "txt", "csv", "md"],
            help="Scanned documents are not supported yet!",
        )

        if file:
            docs = splitting_file(file, st.session_state["CHUNK_SIZE"], st.session_state["CHUNK_OVERLAP"])
            summarize_documents(
                docs,
                cache_flag=file.file_id,
            )
            indexing_documents(
                file_name=file.name,
                embedding_model=st.session_state["EMBED_MODEL_NAME"],
                cache_flag=file.file_id,
                _docs=docs,
            )
        else:
            st.stop()

    # Question Answering Panel
    with st.form(key="qa_form"):
        query = st.text_input("Ask a question about the document")
        submit = st.form_submit_button("Submit")

    if submit:
        with st.spinner("Wait for it...⏳"):
            if not is_query_valid(query):
                st.stop()

            if st.session_state["folder_index"] is not None:
                try:
                    result = query_folder(
                        folder_index=st.session_state["folder_index"],
                        query=query,
                        callback=st.session_state["DEBUG_CALLBACK"],
                        llm=st.session_state["LLM"],
                        top_k=st.session_state["EMBED_TOP_K"],
                        summary_language=st.session_state["SUMMARY_LANGUAGE"],
                        chain_type=st.session_state["CHAIN_TYPE"],
                    )
                    st.balloons()
                    if st.session_state["LANGCHAIN_DEBUG"]:
                        show_debug(st, result.tracks)
                        # Output Columns
                        answer_col, sources_col = st.columns(2)
                        with answer_col:
                            st.markdown("#### Answer")
                            st.markdown(result.answer)
                            st.markdown("#### Sources")
                            for source in result.sources:
                                st.text(f'📄{source.metadata["source"]} {source.page_content}')
                                st.markdown("---")

                        with sources_col:
                            st.markdown("#### Similarity Documents")
                            for doc in result.relevant_docs:
                                st.text(f'📄{doc.metadata["source"]} {doc.page_content}')
                                st.markdown("---")
                    else:
                        st.markdown("#### Answer")
                        st.markdown(result.answer)
                except Exception as e:
                    st.error(e)
                    logger.error(e)


def files_to_docs(files: List[File]) -> List[Document]:
    """Combines all the documents in a list of files into a single list."""

    all_texts = []
    for file in files:
        for doc in file.docs:
            doc.metadata["file_name"] = file.name
            doc.metadata["file_id"] = file.id
            all_texts.append(doc)

    return all_texts
