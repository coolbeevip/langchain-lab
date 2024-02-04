# LangChain LAB | [中文](README_ZH.md)

This experiment will verify different use cases of LLM using LangChain, which include chat, role-playing, and document-based QA.

## Prerequisites

* python 3.11+
* make
* [poetry](https://python-poetry.org/docs/#installation)

## Quick Start

Start the Docker container by binding langchain-lab to external port 8080.

```shell
docker run -d --rm -p 8080:8080 \
-e OPENAI_API_BASE=http://t5.coolbeevip.com:8080 \
-e OPENAI_API_KEY=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 \
coolbeevip/langchain-lab
```
Visit http://localhost:8080 to access the langchain-lab web page.

## Development Guide

Create a `.env` file in the project's root directory and input OpenAI's API key and base url in the file.

```text
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=sk-xx
```

Install the dependencies.

```shell
make install
```

Run the project.

```shell
make run
```

## Screenshot

![](docs/image-chat.png)
![](docs/image-player.png)
![](docs/image-doc.png)

## Environmental Variables

You can configure additional parameters either through environment variables or the `.env` file.

**`OPENAI_API_BASE` / `OPENAI_API_KEY`**

To set the API KEY, use the `OPENAI_API_KEY` variable. Specify the call address by setting the `OPENAI_API_BASE` variable. If desired. Alternatively, you can choose not to set these parameters initially and configure them on the page later.

**`HUGGINGFACE_CATCH_PATH`**

To set the hugging face cache directory, use `HUGGINGFACE_CATCH_PATH`. The default value is `./huggingface`. When using a Compatible  OpenAI API, you can utilize the EMBED huggingface models. After the initial selection, these models will be downloaded to the cache directory. The currently available models are as follows:

* moka-ai/m3e-base
* sentence-transformers/msmarco-distilbert-base-v4
* shibing624/text2vec-base-chinese
