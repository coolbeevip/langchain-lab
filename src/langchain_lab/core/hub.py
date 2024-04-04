import logging
from pathlib import Path
from typing import Any, Optional

from langchain import hub
from langchain_core.load.load import loads


def hub_pull(
    owner_repo_commit: str,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    hub_cache_path = _init_hub_cache()
    prompt_cache_file_name = owner_repo_commit.replace("/", "_").replace(":", "_")
    prompt_cache_file_path = hub_cache_path / prompt_cache_file_name
    prompt = None
    if prompt_cache_file_path.exists():
        try:
            with open(prompt_cache_file_path, "r") as f:
                prompt_hub_config = f.read()
                prompt = loads(prompt_hub_config)
                logging.info(f"Loaded prompt from cache at {prompt_cache_file_path}")
        except Exception as e:
            logging.error(f"Error loading prompt from cache at {prompt_cache_file_path}: {e}")
            prompt = None

    if not prompt:
        client = hub._get_client(api_url=api_url, api_key=api_key)
        prompt_hub_config = client.pull(owner_repo_commit)
        with open(prompt_cache_file_path, "w") as f:
            f.write(prompt_hub_config)
        prompt = loads(prompt_hub_config)
        logging.info(f"Pulled prompt from {owner_repo_commit} and cached it at {prompt_cache_file_path}")

    return prompt


def _init_hub_cache() -> Path:
    user_dir = Path.home()
    hub_cache_path = user_dir / ".hub_cache"
    if not hub_cache_path.exists():
        hub_cache_path.mkdir()
        logging.info(f"Created hub cache directory at {hub_cache_path}")
    return hub_cache_path
