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

from typing import List

from src.langchain_lab.core.llm import TrackItem


def show_debug(st, tracks: List[TrackItem]):
    with st.expander("DEBUG", expanded=True):
        try:
            for track in tracks:
                if track.event == "start":
                    st.info(track.time)
                    st.code(f"{track.message}", line_numbers=True)
                elif track.event == "end":
                    st.success(track.time)
                    st.code(f"{track.message}", line_numbers=True)
                elif track.event == "error":
                    st.error(track.time)
                    st.code(f"{track.message}", line_numbers=True)
        finally:
            st.session_state["DEBUG_CALLBACK"].get_tracks().clear()
