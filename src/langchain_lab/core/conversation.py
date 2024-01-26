"""
Reference https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py

-- Vicuna template --
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: How are you? ASSISTANT: # noqa: E501


-- Llama-2 template --
[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

Hello! [/INST] Hi! </s><s>[INST] How are you? [/INST]


-- ChatGPT template --
[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hello!'}, {'role': 'assistant', 'content': 'Hi!'}, {'role': 'user', 'content': 'How are you?'}] # noqa: E501


-- Claude template --


Human: Hello!

Assistant: Hi!

Human: How are you?

Assistant:

"""
from typing import List


def get_conversation_prompt(name: str, chat_history: List[str]) -> str:
    chat_history_string = ""
    if "vicuna" in name:
        for i, message in enumerate(chat_history):
            if i == 0:
                chat_history_string = message + " "
            else:
                if i % 2 == 0:
                    chat_history_string += message + " "
                else:
                    chat_history_string += message + "</s>"
        return chat_history_string
    else:
        return "\n".join(chat_history)


def get_conversation_human_message_template(name: str):
    return "{question}" + get_step_between_human_and_ai(name) + "ASSISTANT:"


def get_human_prefix(name: str):
    if "vicuna" in name:
        return "USER"
    else:
        return "Human"


def get_ai_prefix(name: str):
    if "vicuna" in name:
        return "ASSISTANT"
    else:
        return "AI"


def get_step_between_human_and_ai(name: str):
    if "vicuna" in name:
        return " "
    else:
        return "\n"
