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
import json
from datetime import datetime
from typing import Any, Dict, List, Union

import requests
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import LLMResult
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI

from langchain_lab import logger


class TrackItem:
    run_id: str
    event: str
    time: str
    message: str

    def __init__(
        self,
        run_id: str,
        event: str,
        icon: str,
        time: datetime,
        message: str,
        tips: str = "",
    ):
        self.run_id = run_id
        self.event = event
        self.time = f'{icon} {time.strftime("%Y-%m-%d %H:%M:%S")} {tips}'
        self.message = message


class TrackerCallbackHandler(BaseCallbackHandler):
    tracks = []
    message_placeholder_message = ""

    def __init__(self, streamlit):
        self.message_placeholder = None
        self.st = streamlit
        logger.info("Initializing TrackerCallbackHandler")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        run_id = str(kwargs["run_id"])
        self.tracks.append(TrackItem(run_id, "start", "🤖", datetime.now(), prompts[0]))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.message_placeholder:
            # print(token)
            self.message_placeholder_message += f"{token}"
            self.message_placeholder.markdown(f"{self.message_placeholder_message}", unsafe_allow_html=True)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = str(kwargs["run_id"])
        if response.llm_output:
            token_usage = response.llm_output["token_usage"]
            prompt_tokens = (token_usage["prompt_tokens"],)
            completion_tokens = (token_usage["completion_tokens"],)
            total_tokens = (token_usage["total_tokens"],)
            token_usage_info = f"💰{prompt_tokens[0]} + 💰 {completion_tokens[0]} = 💰 {total_tokens[0]}"
        else:
            # https://community.openai.com/t/openai-api-get-usage-tokens-in-response-when-set-stream-true/141866/12
            token_usage_info = ""
        for chats in response.generations:
            for chat in chats:
                self.tracks.append(
                    TrackItem(
                        run_id,
                        "end",
                        "🤖",
                        datetime.now(),
                        chat.text,
                        token_usage_info,
                    )
                )

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        self.tracks.append(TrackItem("error", "🤖", datetime.now(), str(error)))

    def get_tracks(self) -> List[TrackItem]:
        return self.tracks.copy()

    def clean_tracks(self):
        self.tracks.clear()

    def init_message_placeholder(self, message_placeholder):
        self.message_placeholder = message_placeholder
        self.message_placeholder_message = ""

    def clean_message_placeholder(self):
        self.message_placeholder = None
        self.message_placeholder_message = ""


@st.cache_resource()
def llm_init(openai_api_base, openai_api_key, model_name, temperature, stream_api) -> BaseChatModel:
    if is_open_ai_key_valid(openai_api_base, openai_api_key, model_name):
        llm = ChatOpenAI(
            model_name=model_name,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            temperature=temperature,
            request_timeout=600,
            streaming=stream_api,
            callbacks=[st.session_state["DEBUG_CALLBACK"]],
        )
        return llm
    else:
        return None


def is_open_ai_key_valid(openai_api_base, openai_api_key, model_name) -> bool:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar and click Refresh!")
        return False
    try:
        client = OpenAI(
            base_url=openai_api_base,
            api_key=openai_api_key,
        )
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
        )
    except Exception as e:
        st.error(e)
        logger.error(f"{e.__class__.__name__}: {e}")
        return False
    return True


def load_llm_chat_models(api_url: str, api_key: str) -> List[str]:
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{api_url}/models", headers=headers)
        print(json.dumps(response.json()))
        model_id_list = [obj["id"] for obj in response.json()["data"]]
        # 如果是 openai 模型，需要过滤掉非 chat 模型
        filtered_models = [s for s in model_id_list if s.startswith("gpt-")]
        if len(filtered_models) == 0:
            filtered_models = model_id_list
        return sorted(filtered_models)
    except Exception as e:
        st.error(f"Failed to load models from {api_url}")
        logger.error(f"{e.__class__.__name__}: {e}")
        return []
