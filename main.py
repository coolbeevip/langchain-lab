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
import os

import streamlit as st

from src.langchain_lab.scenarios.chat import init_chat_scenario
from src.langchain_lab.scenarios.document import init_document_scenario
from src.langchain_lab.scenarios.sidebar import left_sidebar

st.set_page_config(page_title="LangChain LAB", page_icon="📖", layout="wide")

# st.markdown(
#     """<style>
#     .eczjsme4 {
#         padding: 0rem 0rem;
#     }
#     .ea3mdgi4 {
#         padding: 3rem 1rem 10rem;
#     }
#     html {
#         font-size: 12px !important;
#     }
#     p {
#         font-size: 12px !important;
#     }
#     code {
#         font-size: 12px !important;
#     }
#     .stChatFloatingInputContainer {
#         padding-bottom: 10px;
#     }
#     .stHeadingContainer {
#         position: fixed;
#         z-index: 999980;
#         top: 0;
#     }
#     </style>""",
#     unsafe_allow_html=True,
# )  # noqa: W291

st.markdown(
    """<style>
    html {
        font-size: 12px !important;
    }
    p {
        font-size: 12px !important;
    }
    code {
        font-size: 12px !important;
    }
    .eczjsme4 {
        padding: 1rem 1rem;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }    
    </style>""",
    unsafe_allow_html=True,
)  # noqa: W291

if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

left_sidebar()

if 'LLM' not in st.session_state:
    st.warning(
        'Select an AI platform in the sidebar and enter your API address and KEY.'
    )
else:
    # st.header(f"{st.session_state['SCENARIO']}")
    if st.session_state['SCENARIO'] == 'DOCUMENT':
        init_document_scenario()
    elif st.session_state['SCENARIO'] == 'CHAT':
        chat_memory_enabled = st.session_state.get('CHAT_MEMORY_ENABLED', False)
        chat_memory_history_deep = st.session_state.get('CHAT_MEMORY_HISTORY_DEEP', 20)
        chat_stream_api = st.session_state.get('STREAM_API', False)

        init_chat_scenario(
            chat_memory_enabled=chat_memory_enabled,
            chat_memory_history_deep=chat_memory_history_deep,
            chat_stream_api=chat_stream_api
        )
