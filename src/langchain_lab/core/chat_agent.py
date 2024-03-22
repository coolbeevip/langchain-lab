# Copyright 2024 Pengxuan Men
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
import json
from os import path
from pathlib import Path
from typing import Any, Dict, Sequence

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.load.load import Reviver
from langchain_core.tools import BaseTool

from src.langchain_lab.core.llm import TrackerCallbackHandler

default_system_message = """You are a nice chatbot having a conversation with a human.
"""


def chat_agent_once(inputs: Dict[str, Any], llm: BaseChatModel, prompt: PromptTemplate, callback: TrackerCallbackHandler = None):
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callbacks=[callback],
    )
    return chain.invoke(inputs)["text"]


def chat_agent(
    query: str,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    callback: TrackerCallbackHandler = None,
):
    # agent_prompt = hub.pull("hwchase17/openai-functions-agent")
    # https://api.hub.langchain.com/commits/hwchase17/openai-functions-agent/a1655024b06afbd95d17449f21316291e0726f13dcfaf990cc0d18087ad689a5
    agent_prompt_filename = path.join(path.dirname(path.abspath(__file__)), "../hub_offline/hwchase17_openai-functions-agent_a1655024.json")
    agent_prompt_json = json.loads(Path(agent_prompt_filename).read_text())
    agent_prompt = json.loads(json.dumps(agent_prompt_json["manifest"]), object_hook=Reviver())

    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=agent_prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, callback=callback, verbose=True)

    result = agent_executor.invoke({"input": query})

    return result["output"]
