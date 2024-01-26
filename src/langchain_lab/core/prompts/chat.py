from typing import Any, List, Sequence, Set

from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    BaseChatPromptTemplate,
    BaseMessagePromptTemplate,
    MessageLikeRepresentation,
    _convert_to_message,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    PromptValue,
    SystemMessage,
)

from langchain_lab.core.conversation import (
    get_ai_prefix,
    get_human_prefix,
    get_step_between_human_and_ai,
)


def get_buffer_string(messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI", separator: str = "\n") -> str:
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        message = f"{role}: {m.content}"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return separator.join(string_messages)


class CustomChatPromptValue(PromptValue):
    llm: BaseChatModel
    messages: List[BaseMessage]

    def to_string(self) -> str:
        return get_buffer_string(
            self.messages,
            human_prefix=get_human_prefix(self.llm.model_name),
            ai_prefix=get_ai_prefix(self.llm.model_name),
            separator=get_step_between_human_and_ai(self.llm.model_name),
        )

    def to_messages(self) -> List[BaseMessage]:
        return self.messages


class CustomChatPromptTemplate(ChatPromptTemplate):
    llm: BaseChatModel

    @classmethod
    def from_messages(
        cls,
        llm: BaseChatModel,
        messages: Sequence[MessageLikeRepresentation],
    ) -> ChatPromptTemplate:
        _messages = [_convert_to_message(message) for message in messages]
        input_vars: Set[str] = set()
        for _message in _messages:
            if isinstance(_message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)):
                input_vars.update(_message.input_variables)

        return cls(input_variables=sorted(input_vars), messages=_messages, llm=llm)

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        messages = self.format_messages(**kwargs)
        return CustomChatPromptValue(messages=messages, llm=self.llm)
