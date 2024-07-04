from typing import Any, Dict

from langchain_lab.core.agents import LabAgent


class LangGraphDemoAgent(LabAgent):
    description = "这是一个演示 LangGraph 的 Agent"

    def agent_invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"input": inputs, "output": "hello world"}
