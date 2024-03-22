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

from datetime import datetime

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental import text_splitter

from langchain_lab import logger
from langchain_lab.core.summary import summarize
from langchain_lab.scenarios.error import display_error
from src.langchain_lab.core.chunking import chunk_file
from src.langchain_lab.core.embedding import embed_files
from src.langchain_lab.core.parsing import File, read_file
from src.langchain_lab.core.qa import query_folder
from src.langchain_lab.scenarios.debug import show_debug


@st.cache_resource
def splitting_file(uploaded_file, chunk_size, chunk_overlap):
    __file = read_file(uploaded_file)
    if not is_file_valid(__file):
        st.stop()
    start_time = datetime.now()
    with st.spinner(f"Spitting document {uploaded_file.name}. This may take a while⏳"):
        __chunked_file = chunk_file(__file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        with st.expander(f"Document {len(__chunked_file.docs)}"):
            for index, doc in enumerate(__chunked_file.docs):
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
            f"Document successfully divided into **{len(__chunked_file.docs)}** sections, \
            each with a size of **{chunk_size}**, featuring an overlap of **{chunk_overlap}** ({seconds_diff}s)."
        )
        return __chunked_file


@st.cache_resource
def indexing_documents(file_name: str, docs_size: int, embedding_model, __chunked_file):
    try:
        start_time = datetime.now()
        with st.spinner(f"Indexing **{file_name}** This may take a while⏳"):
            folder_index = embed_files(
                files=[__chunked_file],
                vector_store="faiss",
                embedding=st.session_state["EMBEDDING"],
            )
            st.session_state["folder_index"] = folder_index
            end_time = datetime.now()
            time_diff = end_time - start_time
            seconds_diff = time_diff.total_seconds()
            st.info(f"Completed **{len(__chunked_file.docs)}** sections indexes by **{embedding_model}**! ({seconds_diff}s)")

    except Exception as e:
        display_error(logger, e)


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
    logger.info("Initializing document scenario")
    upload_tab, web_tab = st.tabs(["Upload", "Web"])

    with web_tab:
        with st.form(key="web_loader_form"):
            url_input = st.text_input("Enter a url", placeholder="https://www.example.com")
            load_btn = st.form_submit_button("Load Documents")
        if load_btn:
            print(url_input)
            loader = WebBaseLoader(url_input)
            data = loader.load()
            splits = text_splitter.split_documents(data)
            st.write(splits)
            # chunked_file = splitting_file(file, st.session_state["CHUNK_SIZE"], st.session_state["CHUNK_OVERLAP"])
            # indexing_documents(
            #     file.name,
            #     len(chunked_file.docs),
            #     st.session_state["EMBED_MODEL_NAME"],
            #     chunked_file,
            # )

    with upload_tab:
        file = st.file_uploader(
            "Upload a pdf, docx, txt or csv file",
            type=["pdf", "docx", "txt", "csv"],
            help="Scanned documents are not supported yet!",
        )

        if not file:
            st.stop()
        else:
            chunked_file = splitting_file(file, st.session_state["CHUNK_SIZE"], st.session_state["CHUNK_OVERLAP"])
            indexing_documents(
                file.name,
                len(chunked_file.docs),
                st.session_state["EMBED_MODEL_NAME"],
                chunked_file,
            )

    # Summary Panel
    with st.form(key="summary_form"):
        summary_submit = st.form_submit_button("Summarize")
    if summary_submit:
        with st.spinner("Wait for summarize...⏳"):
            response = summarize(docs=chunked_file.docs, llm=st.session_state["LLM"], callback=st.session_state["DEBUG_CALLBACK"])
            st.markdown(response)

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
