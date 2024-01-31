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

from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from langchain_lab import logger
from langchain_lab.core.llm import TrackerCallbackHandler

summary_template = """Please use the following topics or keywords to generate an outline that includes titles, chapters, and subsections. Output it in Markdown format. Only give me the output and nothing else. The outline should be in the ${lang} language. Topics or keywords: \"\"\"${selection}\"\"\"
"""


def summarize(
    docs: List[Document],
    llm: BaseChatModel,
    callback: TrackerCallbackHandler = None,
):
    summary_prompt = PromptTemplate.from_template(summary_template)
    selections = ""
    doc_idx = 0
    for doc in docs:
        selections = selections + doc.page_content
        if len(selections) > 2000:
            inputs = {"lang": "Chinese", "selection": selections}
            chain = LLMChain(
                llm=llm,
                prompt=summary_prompt,
                callbacks=[callback],
            )
            summary = chain.invoke(inputs)["text"]
            logger.info(f"Summarize: {doc_idx}/{len(docs)}")
            logger.info(f"Prompt: {selections}")
            logger.info(f"Summary: {summary}")
            selections = summary + "\n"
        doc_idx = doc_idx + 1
    return selections
