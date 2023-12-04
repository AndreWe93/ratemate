import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.scrape_apify import scrape_apify
from interface.main import *
from params import *
from ml_logic.NLP import new_column_NLP
from ml_logic.calculate_final_score import *

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#app.state.model = load_model()

@app.get("/personal_score")
def predict(
    url: str,
    price_review_weightage: float = Query(..., ge=0, le=1),
    food_review_weightage: float = Query(..., ge=0, le=1),
    service_review_weightage: float = Query(..., ge=0, le=1),
    ambience_review_weightage: float = Query(..., ge=0, le=1),
    local_guides_review_weightage: bool = Query(...),
):
    """
    """
    url = url
    df = scrape_apify(url)
    df = df[COLUMN_NAMES_RAW]
    pre_processed_df = preprocess_reviews_text(df) # Still need to do the column selection

    # Classification of reviews
    #classified_df = classify_reviews_df(pre_processed_df, "reviews_without_SW")

    # Get subratings
    # Load the production model and calculate subratings (without the price subrating)
    # Fill n the price subrating after figuring it out
    # In the end don't use fill_sub_ratings function

    # model = app.state.model
    # assert model is not None

    subratings_df = new_column_NLP(pre_processed_df)

    #subratings_df = create_sub_ratings(classified_df) # This is a place holder
    subratings_df_price = df_with_price_rating(subratings_df)

    # Average scores
    average_scores_df = calculate_average_scores(subratings_df_price, price_review_weightage, service_review_weightage, ambience_review_weightage, food_review_weightage)

    # Overall score
    overall_score = calculate_overall_score(average_scores_df)

    return f"Your personal score is: {overall_score}"

@app.get("/")
def root():
    return {
    'greeting': "Hello friend, let's get this party started!"
}
