FROM python:3.10.6-buster

WORKDIR /ratemate_01
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader stopwords

COPY api api
COPY interface interface
COPY ml_logic ml_logic
COPY setup.py setup.py
COPY params.py params.py
COPY tokenizer.pkl tokenizer.pkl
COPY tfidf_vectorizer.pkl tfidf_vectorizer.pkl

RUN pip install .


CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
