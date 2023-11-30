FROM python:3.10.6-buster

WORKDIR /ratemate_app_test
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY api api
COPY ml_logic ml_logic
COPY setup.py setup.py

RUN pip install .


CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
