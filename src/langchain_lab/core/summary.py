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

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from langchain_lab.core.llm import TrackerCallbackHandler

question_prompt_template = """
                  Please provide a concise {lang} summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """

refine_prompt_template = """
              Write a concise {lang} summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              Please keep in mind that the maximum length allowed for your text is 300 words.
              BULLET POINT SUMMARY:
              """


# refine_prompt_template = (
#     "Your job is to produce a final concise summary\n"
#     "We have provided an existing summary up to a certain point: {existing_answer}\n"
#     "We have the opportunity to refine the existing summary"
#     "(only if needed) with some more context below.\n"
#     "------------\n"
#     "{text}\n"
#     "------------\n"
#     "Given the new context, refine the original summary in {lang}"
#     "If the context isn't useful, return the original summary."
# )


def summarize(
    docs: List[Document],
    llm: BaseChatModel,
    summary_language: str = "English",
    callback: TrackerCallbackHandler = None,
):
    question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["text", "lang"])
    refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["text", "lang"])

    refine_chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
    )
    refine_outputs = refine_chain({"lang": summary_language, "input_documents": docs})
    final_refine_data = []
    for out in refine_outputs["intermediate_steps"]:
        final_refine_data.append(out)
    # summary_prompt = PromptTemplate.from_template(summary_template)
    # selections = ""
    # doc_idx = 0
    # for doc in docs:
    #     selections = selections + doc.page_content
    #     if len(selections) > 2000:
    #         inputs = {"lang": "Chinese", "selection": selections}
    #         chain = LLMChain(
    #             llm=llm,
    #             prompt=summary_prompt,
    #             callbacks=[callback],
    #         )
    #         summary = chain.invoke(inputs)["text"]
    #         logger.info(f"Summarize: {doc_idx}/{len(docs)}")
    #         logger.info(f"Prompt: {selections}")
    #         logger.info(f"Summary: {summary}")
    #         selections = summary + "\n"
    #     doc_idx = doc_idx + 1
    return "\n".join(final_refine_data)
