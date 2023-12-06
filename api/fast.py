from params import *
from interface.main import *

from ml_logic.calculate_final_score import *
from ml_logic.scrape_apify import scrape_apify
from ml_logic.random_forest_model import pred_from_random_forest

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/personal_score")
def predict(
    url: str,
    price_review_weightage: float = Query(..., ge=0, le=1),
    food_review_weightage: float = Query(..., ge=0, le=1),
    service_review_weightage: float = Query(..., ge=0, le=1),
    ambience_review_weightage: float = Query(..., ge=0, le=1),
):
    """
    """
    url = url
    df = scrape_apify(url)
    df = df[COLUMN_NAMES_RAW]
    pre_processed_df = preprocess_reviews_text(df) # Still need to do the column selection

    # Classification of reviews
    classified_df = classify_reviews_df(pre_processed_df, "reviews_without_SW")

    # Get subratings
    subratings_df = pred_from_random_forest(classified_df)

    # Get subratings for price
    subratings_df_price = df_with_price_rating(subratings_df)

    # Average scores
    average_scores_df = calculate_average_scores(subratings_df_price, price_review_weightage, service_review_weightage, ambience_review_weightage, food_review_weightage)

    # Overall score
    personal_score = calculate_overall_score(average_scores_df)

    return f"RateMate Rating: ⭐️ {personal_score} ⭐️"

@app.get("/")
def root():
    return {
    'greeting': "Hello friend, let's get this party started!"
}
