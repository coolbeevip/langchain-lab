

lint:
	@poetry run black src
	@poetry run isort --profile black src
	@poetry run flake8 src

install:
	@poetry lock
	@poetry install

run:
	@streamlit run main.py
