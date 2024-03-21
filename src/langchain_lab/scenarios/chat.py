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

import streamlit as st
from langchain.prompts import PromptTemplate

from langchain_lab import logger
from langchain_lab.agent_tools.weather.weather import weather
from src.langchain_lab.core.chat import chat, chat_once
from src.langchain_lab.core.chat_agent import chat_agent, chat_agent_once
from src.langchain_lab.scenarios.debug import show_debug


def chat_scenario(
    query: str,
    message_placeholder,
    chat_history: List[str] = None,
    chat_stream_api: bool = False,
    chat_memory_history_deep: int = 20,
    with_agent: bool = False,
):
    full_response = ""

    tools = []
    if with_agent:
        if st.session_state["WEATHER_TOOL"]:
            tools.append(weather)

    if len(tools) > 0:
        response = chat_agent(
            query=query,
            callback=st.session_state["DEBUG_CALLBACK"],
            llm=st.session_state["LLM"],
            tools=tools,
        )
    else:
        response = chat(
            query=query,
            callback=st.session_state["DEBUG_CALLBACK"],
            llm=st.session_state["LLM"],
            system_message=st.session_state.get("CHAT_PROMPT_TEMPLATE", ""),
            chat_history=chat_history,
        )
    for chunk in response.split("\n"):
        full_response += chunk + " "

    if not chat_stream_api:
        message_placeholder.markdown(full_response)

    if st.session_state["CHAT_MEMORY_ENABLED"]:
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": query, "avatar": "🧑‍💻"})
        # Add assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response, "avatar": "🤖"})
        # Limit chat history length
        st.session_state.chat_messages = st.session_state.chat_messages[-chat_memory_history_deep:]


def summarize_human_questions_by_chat_history(chat_history: List[str] = None, with_agent: bool = False) -> str:
    prompt_template = """请根据以下对话,一句话整理Human提供的信息:
    "{text}"
    """
    prompt = PromptTemplate.from_template(prompt_template)
    if with_agent:
        response = chat_agent_once(inputs={"text": "\n".join(chat_history)}, callback=st.session_state["DEBUG_CALLBACK"], llm=st.session_state["LLM"], prompt=prompt)
    else:
        response = chat_once(inputs={"text": "\n".join(chat_history)}, callback=st.session_state["DEBUG_CALLBACK"], llm=st.session_state["LLM"], prompt=prompt)
    return response


def init_chat_scenario(
    chat_memory_enabled: bool = False,
    chat_memory_history_deep: int = 20,
    chat_stream_api: bool = False,
    with_agent: bool = False,
):
    logger.info(
        "Initializing chat scenario with memory={memory} deep={deep}".format(
            memory=chat_memory_enabled,
            deep=chat_memory_history_deep,
        )
    )
    prompt_placeholder = st.empty()
    if "CHAT_PROMPT_TEMPLATE" in st.session_state:
        prompt_string = st.session_state["CHAT_PROMPT_TEMPLATE"]
        if len(prompt_string.strip()) > 0:
            # prompt_string = (prompt_string
            #                  .replace("-", "\\-")
            #                  .replace("\n", "\n\n")
            #                  .replace("{", "\\{")
            #                  .replace("}", "\\}"))
            prompt_placeholder.info(f"{prompt_string[:50 - 3]}...")

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat messages from history on app rerun
    for chat_messages in st.session_state.chat_messages:
        with st.chat_message(name=chat_messages["role"], avatar=chat_messages["avatar"]):
            st.markdown(chat_messages["content"])

    # Display chat input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message(name="user", avatar="🧑‍💻"):
            st.markdown(prompt)
        chat_history = None
        if chat_memory_enabled:
            chat_history = [f"{'USER' if message['role'] == 'user' else 'ASSISTANT'}: {message['content']}" for message in st.session_state.chat_messages]

        with st.chat_message(name="assistant", avatar="🤖"):
            message_placeholder = st.empty()
            message_placeholder.markdown("...")
            st.session_state["DEBUG_CALLBACK"].init_message_placeholder(message_placeholder)
            try:
                chat_scenario(prompt, message_placeholder, chat_history, chat_stream_api, chat_memory_history_deep,
                              with_agent)
            finally:
                st.session_state["DEBUG_CALLBACK"].clean_message_placeholder()

    if st.session_state["LANGCHAIN_DEBUG"]:
        show_debug(st, st.session_state["DEBUG_CALLBACK"].get_tracks())
