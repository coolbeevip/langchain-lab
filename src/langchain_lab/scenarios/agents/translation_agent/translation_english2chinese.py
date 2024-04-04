import os
from typing import Any, Dict

import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

from langchain_lab.core.agents import LabAgent


class TranslationEnglish2Chinese(LabAgent):
    description = (
        '这是一个英文翻译中文的代理，参考了宝玉老师《怎么让ChatGPT的翻译结果更准确》https://twitter.com/dotey/status/1711494319465496656。你可以试着让他翻译:\nThe "AI girlfriend" idea is a tar pit - avoid it.'  # noqa: E501
    )

    def __init__(self):
        system_prompt_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.txt")
        with open(system_prompt_file_path, "r") as f:
            self.system_prompt_template = f.read()

    def agent_invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_template),
                ("user", "{input}"),
            ]
        )
        llm = st.session_state["LLM"]
        chain = LLMChain(
            llm=llm,
            prompt=chat_prompt,
        )
        return {"input": inputs, "output": chain.invoke(inputs)["text"]}
