lint:
	@poetry run black src
	@poetry run isort --profile black src
	@poetry run flake8 src

fmt: lint
	@black ./src
	@isort --profile black ./src

install:
	@poetry lock
	@poetry install

run:
	@streamlit run main.py
