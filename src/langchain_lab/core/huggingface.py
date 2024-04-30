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

from sentence_transformers import SentenceTransformer


def download_hugging_face_model(model_name):
    cache_folder = os.environ["HUGGINGFACE_CATCH_PATH"]
    SentenceTransformer(os.path.join(cache_folder, model_name), cache_folder=cache_folder)
