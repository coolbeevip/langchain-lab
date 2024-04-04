FROM coolbeevip/langchain-cpu:3.11-0.1.14 as builder
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
WORKDIR /usr/app/langchain-lab
RUN python -m venv /usr/app/venv
ENV PATH="/usr/app/venv/bin:$PATH"
COPY pyproject.toml .
COPY poetry.lock .
COPY src src
COPY main.py main.py
COPY README.md README.md
COPY .streamlit .streamlit
RUN pip install .

FROM python:3.11-slim as production
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
WORKDIR /usr/app/langchain-lab
COPY --from=builder /usr/app/venv ../venv
COPY --from=builder /usr/app/langchain-lab/src src
COPY --from=builder /usr/app/langchain-lab/main.py main.py
COPY --from=builder /usr/app/langchain-lab/.streamlit .streamlit

EXPOSE 8080
ENV PYTHONPATH=${PYTHONPATH}:.
ENV PATH="/usr/app/venv/bin:$PATH"
CMD [ "streamlit", "run", "main.py" ]
