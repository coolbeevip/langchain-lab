from langchain_lab.core.agents import agent_register
from langchain_lab.scenarios.agents.wikipedia_agent.wikipedia_agent import (
    WikipediaAgent,
)

agent_register(WikipediaAgent())
