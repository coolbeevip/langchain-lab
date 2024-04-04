from typing import Any, Dict

import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

from langchain_lab.core.agents import LabAgent
from langchain_lab.core.hub import hub_pull


class WikipediaAgent(LabAgent):
    description = "WikipediaAgent is an AI assistant that delivers quick, relevant answers from Wikipedia on a wide range of topics, offering a reliable resource for information and learning."  # noqa: E501

    def agent_invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        prompt = hub_pull("hwchase17/openai-tools-agent")
        model = st.session_state["LLM"]
        tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]
        agent = create_openai_tools_agent(model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        return agent_executor.invoke(input)
