import streamlit as st

from ml_logic.random_forest_model import pred_from_random_forest
from ml_logic.calculate_final_score import *

from interface.main import *
from params import *

import requests
from ml_logic.scrape_apify import scrape_apify

# ####### TO Dos #########
# """
# 1. Show the original score
# 2. Show the subratings from the dataset
# 3. Integrate the Word cloud picture from my notebook somehow into the streamlit app
# 4. Add more features
# """
st.snow()

st.markdown("""# RateMate
## Get your personal rating for the restaurant of your choice
Hello friend, please insert the google maps url of the restaurant you are interested in:""")

# get the user input
restaurant_url = st.text_input("url of your restaurant")

import streamlit as st

st.title("Adjust Sliders (Sum to 1.0)")

# Function to create sliders with a constraint on the sum
def create_sliders(num_sliders):
    # Create sliders with initial values
    sliders = [st.slider(f"Slider {i + 1}", 0.0, 1.0, value=1.0 / num_sliders) for i in range(num_sliders)]

    # Calculate the sum of slider values
    total_sum = sum(sliders)

    # Normalize sliders to ensure the sum is 1.0
    normalized_sliders = [slider / total_sum for slider in sliders]

    return normalized_sliders

# Number of sliders
num_sliders = 3

# Create sliders with a constraint on the sum
sliders = create_sliders(num_sliders)

# Display the normalized slider values
st.write("Normalized Slider Values:", sliders)


# the selected value is returned by st.slider
price_review_weightage = st.slider('Select your price rating weight', 0.0, 1.0, 0.25)
food_review_weightage = st.slider('Select your food rating weight', 0.0, 1.0, 0.25)
service_review_weightage = st.slider('Select your service rating weight', 0.0, 1.0, 0.25)
ambience_review_weightage = st.slider('Select your ambience rating weight', 0.0, 1.0, 0.25)

# checkbox for local guides
local_guides_review_weightage = st.checkbox('Review only from local guides')

# Save user input to session state
st.session_state.restaurant_url = restaurant_url


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
    # df = scrape_apify(url)
    df = pd.read_csv("./raw_data_slim/merged_slim_file.csv")
    df = df.head(1000).copy()
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

    # Original score
    original_score = round(df.stars.mean(), 2)

    st.divider()
    st.header(f'Restuarant Rating: ⭐️ {original_score} ⭐️')

    st.header(f'Your Personal Score: ⭐️ {personal_score} ⭐️')

# Create a link to navigate to the next page
if st.button("Reviews Overview"):
    st.session_state.page = "Reviews_Overview"
