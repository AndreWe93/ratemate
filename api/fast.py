import pandas as pd
# pip install fastapi
# pip install uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.scrape_apify import scrape_apify

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/scrape")
def scrape(url):

    url = url
    df = scrape_apify(url)

    return {"average score of ratings": df.stars.mean()}


@app.get("/")
def root():
    return {
    'greeting': "Hello friend, let's get this party started!"
}
