from abc import ABC, abstractmethod
from typing import Any, Dict


class LabAgent(ABC):
    description: str

    @abstractmethod
    def agent_invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.agent_invoke(input)["output"]


registered_agents: dict[str, LabAgent] = {}


def get_agent_list():
    return registered_agents


def get_agent_by_name(agent_name: str):
    return registered_agents.get(agent_name)


def agent_register(agent: LabAgent):
    registered_agents[agent.__class__.__name__] = agent
