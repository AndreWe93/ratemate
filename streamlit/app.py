import streamlit as st
import requests

from ml_logic.scrape_apify import scrape_apify
from ml_logic.NLP import new_column_NLP
from ml_logic.calculate_final_score import *
from interface.main import *
from params import *

st.markdown("""# RateMate
## Get your personal rating for the restaurant of your choice
Hello friend, please insert the google maps url of the restaurant you are interested in:""")

# get the user input
restaurant_url = st.text_input("url of your restaurant")

# the selected value is returned by st.slider
price_review_weightage = st.slider('Select your price rating weight', 0.0, 1.0, 0.25)
food_review_weightage = st.slider('Select your food rating weight', 0.0, 1.0, 0.25)
service_review_weightage = st.slider('Select your service rating weight', 0.0, 1.0, 0.25)
ambience_review_weightage = st.slider('Select your ambience rating weight', 0.0, 1.0, 0.25)

# checkbox for local guides
local_guides_review_weightage = st.checkbox('Review only from local guides')


if st.button("Get Score"):
    #url = 'https://awtestratemate2-z2kqlvo2ta-ew.a.run.app/scrape'

    params = {
            'url': restaurant_url
            }
    url = restaurant_url
    price = price_review_weightage
    food = food_review_weightage
    service = service_review_weightage
    ambience = ambience_review_weightage

    # Make API request
    #response = requests.get(url, params=params)
    df = scrape_apify(url)
    df = df[COLUMN_NAMES_RAW]
    pre_processed_df = preprocess_reviews_text(df) # Still need to do the column selection

    # Classification of reviews
    classified_df = classify_reviews_df(pre_processed_df, "reviews_without_SW")

    # Get subratings
    # Load the production model and calculate subratings (without the price subrating)
    # Fill n the price subrating after figuring it out
    # In the end don't use fill_sub_ratings function

    # model = app.state.model
    # assert model is not None
    subratings_df = new_column_NLP(classified_df)

    #subratings_df = create_sub_ratings(classified_df) # This is a place holder
    subratings_df_price = df_with_price_rating(subratings_df)

    # Average scores
    average_scores_df = calculate_average_scores(subratings_df_price, price, service, ambience, food)

    # Overall score
    overall_score = calculate_overall_score(average_scores_df)

    st.text(overall_score)

    # if response.status_code == 200:
    #     # Retrieve prediction from JSON response
    #     score = response.json()["average score of ratings"]

    #     # Display prediction to the user
    #     st.success(f'The score of your restaurant is: {score}')
    # else:
    #     st.error('Failed to get score from the API. Please try again.')

# ############## copy from streamlit/frontend.py
# # pip install streamlit
# import streamlit as st
# import requests


# '''
# # Let's test our API with some scraping and scoring
# '''

# st.markdown('''
# Hello friend, please insert the google maps url of the restaurant you are interested in:
# ''')


# url_user = st.text_input("url of your restaurant")



# if st.button("Get Score"):
#     url = 'https://awtestratemate2-z2kqlvo2ta-ew.a.run.app/scrape'

#     params = {
#             'url': url_user
#             }

#     # Make API request
#     response = requests.get(url, params=params)

#     if response.status_code == 200:
#         # Retrieve prediction from JSON response
#         score = response.json()["average score of ratings"]

#         # Display prediction to the user
#         st.success(f'The score of your restaurant is: {score}')
#     else:
#         st.error('Failed to get score from the API. Please try again.')
