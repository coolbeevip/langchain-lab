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
import os
from typing import List, Type

import streamlit as st
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from langchain_lab import logger
from src.langchain_lab.core.parsing import File


class FolderIndex:
    """Index for a collection of files (a folder)"""

    def __init__(self, docs: List[Document], index: VectorStore):
        self.name: str = "default"
        self.docs = docs
        self.index: VectorStore = index

    @staticmethod
    def _combine_files(files: List[File]) -> List[Document]:
        """Combines all the documents in a list of files into a single list."""

        all_texts = []
        for file in files:
            for doc in file.docs:
                doc.metadata["file_name"] = file.name
                doc.metadata["file_id"] = file.id
                all_texts.append(doc)

        return all_texts

    @classmethod
    def from_files(cls, files: List[File], embeddings: Embeddings, vector_store: Type[VectorStore]) -> "FolderIndex":
        """Creates an index from files."""
        all_docs = cls._combine_files(files)

        index = vector_store.from_documents(
            documents=all_docs,
            embedding=embeddings,
        )
        return cls(docs=all_docs, index=index)

    @classmethod
    def from_web(cls, url: str, embeddings: Embeddings, vector_store: Type[VectorStore]) -> "FolderIndex":
        loader = WebBaseLoader(url)
        docs = loader.load()
        index = vector_store.from_documents(
            documents=docs,
            embedding=embeddings,
        )

        return cls(docs=docs, index=index)


@st.cache_resource
def embedding_init(provider: str, api_url: str, api_key: str, model_name: str, model_kwargs):
    if provider == "openai":
        embedding = OpenAIEmbeddings(model=model_name, openai_api_base=api_url, openai_api_key=api_key)
    elif provider == "huggingface":
        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=os.environ["HUGGINGFACE_CATCH_PATH"],
            model_kwargs=model_kwargs,
        )
        # embedding.client.max_seq_length = 512
    st.session_state["EMBEDDING"] = embedding
    logger.info(f"Initializing embedding with {model_name}")


def embed_files(files: List[File], embedding, vector_store: str, **kwargs) -> FolderIndex:
    """Embeds a collection of files and stores them in a FolderIndex."""
    supported_vector_stores: dict[str, Type[VectorStore]] = {"faiss": FAISS}

    if vector_store in supported_vector_stores:
        _vector_store = supported_vector_stores[vector_store]
    else:
        raise NotImplementedError(f"Vector store {vector_store} not supported.")

    return FolderIndex.from_files(
        files=files,
        embeddings=embedding,
        vector_store=_vector_store,
    )
