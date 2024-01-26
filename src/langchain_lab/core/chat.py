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

from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import HumanMessagePromptTemplate, PromptTemplate
from langchain.schema import SystemMessage

from langchain_lab.core.conversation import (
    get_conversation_human_message_template,
    get_conversation_prompt,
)
from langchain_lab.core.prompts.chat import CustomChatPromptTemplate
from src.langchain_lab.core.llm import TrackerCallbackHandler

default_system_message = """You are a nice chatbot having a conversation with a human.
"""


def chat_once(inputs: Dict[str, Any], llm: BaseChatModel, prompt: PromptTemplate, callback: TrackerCallbackHandler = None):
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callbacks=[callback],
    )
    return chain.invoke(inputs)["text"]


def chat(
    query: str,
    llm: BaseChatModel,
    system_message: str = None,
    callback: TrackerCallbackHandler = None,
    chat_history: List = None,
):
    if system_message is None or len(system_message) == 0:
        system_message = default_system_message

    if chat_history is not None:
        human_message_prompt = HumanMessagePromptTemplate.from_template(get_conversation_human_message_template(name=llm.model_name))

        # Append chat history to system message if chat history is not empty
        if "{chat_history}" not in system_message:
            system_message = system_message + "\n{chat_history}"
        if len(chat_history) > 0:
            chat_history_string = get_conversation_prompt(name=llm.model_name, chat_history=chat_history)
        else:
            chat_history_string = ""
        system_message_prompt = SystemMessage(content=system_message.format(chat_history=chat_history_string))
    else:
        template = "{question}"
        system_message_prompt = SystemMessage(content=system_message)
        human_message_prompt = HumanMessagePromptTemplate.from_template(template)

    inputs = {"question": query}
    CustomChatPromptTemplate.llm = llm
    chat_prompt = CustomChatPromptTemplate.from_messages(llm=llm, messages=[system_message_prompt, human_message_prompt])

    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        callbacks=[callback],
    )
    return chain.invoke(inputs)["text"]
