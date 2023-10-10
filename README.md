This is an experimental project used to verify various use cases of LLM based on LangChain, including chat, role-playing, and document-based QA.

## Prerequisites

* python 3.10+
* make

## Quick Guide

Create a `.env` file in the root directory of the project and configure OpenAI's API Key and Model Name list in the file.

```text
OPENAI_API_KEY=sk-xxx
OPENAI_API_MODEL_NAMES=gpt-3.5-turbo-16k,gpt-4
```

Install the dependencies.

```shell
make install
```

Run the project.

```shell
streamlit run main.py
```

## Screenshot

![](docs/image-chat.png)
![](docs/image-player.png)
![](docs/image-doc.png)

## Environmental Variables

You can set more parameters through environment variables or the `.env` file.

**`DEFAULT_AI_PLATFORM` / `DEFAULT_AI_PLATFORM_SUPPORT`**

You can set the supported platform through `DEFAULT_AI_PLATFORM_SUPPORT`, with the default value being `OpenAI`. You can set it to `FastChat,OpenAI` to enable calls to the FastChat platform. You can specify the default platform name from the list by using the `DEFAULT_AI_PLATFORM` parameter, with the default value being `OpenAI`.

**`OPENAI_API_BASE` / `OPENAI_API_KEY` / `OPENAI_API_MODEL_NAMES`**

You can set the API KEY through `OPENAI_API_KEY`. Specify the call address by setting `OPENAI_API_BASE`. Set the optional model names through `OPENAI_API_MODEL_NAMES`, separated by commas. The default is `gpt-3.5-turbo-16k,gpt-4`. Of course, you can also choose not to set these parameters and set them on the page after startup.

**`FASTCHAT_API_BASE` / `FASTCHAT_API_KEY` / `FASTCHAT_API_MODEL_NAMES`**

If you have enabled FastChat platform support through `DEFAULT_AI_PLATFORM_SUPPORT`, you can set the API KEY through `FASTCHAT_API_KEY`. Specify the call address by setting `FASTCHAT_API_BASE`. Set the optional model names through `FASTCHAT_API_MODEL_NAMES`. Just like the OpenAI parameters, you can also choose not to set these parameters and set them on the page after startup, but the `FASTCHAT_API_MODEL_NAMES` parameter must provide an optional model name.

**`HUGGINGFACE_CATCH_PATH`**

Set the huggingface cache directory, with the default value being `./huggingface`. When you choose a non-OpenAI API, you can choose to use the EMBED models of huggingface. These models will be downloaded to this cache directory after the first selection. The currently available models are as follows:

* moka-ai/m3e-base
* sentence-transformers/msmarco-distilbert-base-v4
* shibing624/text2vec-base-chinese
