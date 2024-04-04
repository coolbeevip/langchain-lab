from langchain_lab.core.agents import agent_register
from langchain_lab.scenarios.agents.translation_agent.translation_english2chinese import (
    TranslationEnglish2Chinese,
)
from langchain_lab.scenarios.agents.wikipedia_agent.wikipedia_agent import (
    WikipediaAgent,
)

agent_register(WikipediaAgent())
agent_register(TranslationEnglish2Chinese())
