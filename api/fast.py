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

    df_nan = df[df['text'].notna()]
    sorted_df = df_nan.sort_values(by='reviewerNumberOfReviews', ascending=False)
    top3_reviews = list(sorted_df.head(3)['text'])

    df = df[COLUMN_NAMES_RAW]
    pre_processed_df = preprocess_reviews_text(df) # Still need to do the column selection
    wordcloud_input = " ".join(pre_processed_df["reviews_without_SW"].astype(str))

    # Classification of reviews
    classified_df = classify_reviews_df(pre_processed_df, "reviews_without_SW")

    dist_classes = class_dist(classified_df)

    # Get subratings
    subratings_df = pred_from_random_forest(classified_df)

    # Get subratings for price
    subratings_df_price = df_with_price_rating(subratings_df)

    # Average scores
    average_scores_df = calculate_average_scores(subratings_df_price, price_review_weightage, service_review_weightage, ambience_review_weightage, food_review_weightage)

    # Overall score
    personal_score = calculate_overall_score(average_scores_df)
    sub_ratings = individual_scores(average_scores_df)

    #return f"RateMate Rating: ⭐️ {personal_score} ⭐️"
    return {"personal_score": personal_score,
                "top_1": top3_reviews[0],
                "top_2": top3_reviews[1],
                "top_3": top3_reviews[2],
                "sub_price": sub_ratings[0],
                "sub_service": sub_ratings[1],
                "sub_atmosphere": sub_ratings[2],
                "sub_food": sub_ratings[3],
                "dist_price": dist_classes[0],
                "dist_service": dist_classes[1],
                "dist_atmosphere": dist_classes[2],
                "dist_food": dist_classes[3],
                "wordcloud_input": wordcloud_input
                }

@app.get("/")
def root():
    return {
    'greeting': "Hello friend, let's get this party started!"
}
