import streamlit as st
import requests


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


with st.form(key='params_for_api'):

    # get the user input
    restaurant_url = st.text_input("url of your restaurant")

    # the selected value is returned by st.slider
    price_review_weightage = st.slider('Select your price rating weight', 0.0, 1.0, 0.25)
    food_review_weightage = st.slider('Select your food rating weight', 0.0, 1.0, 0.25)
    service_review_weightage = st.slider('Select your service rating weight', 0.0, 1.0, 0.25)
    ambience_review_weightage = st.slider('Select your ambience rating weight', 0.0, 1.0, 0.25)

    # checkbox for local guides
    local_guides_review_weightage = st.checkbox('Review only from local guides')

    st.form_submit_button('Get your Score')

params = dict(
    url = restaurant_url,
    price = price_review_weightage,
    food = food_review_weightage,
    service = service_review_weightage,
    ambience = ambience_review_weightage,
    local_guide = local_guides_review_weightage
    )

ratemate_api_url = 'https://ratemate-z2kqlvo2ta-ew.a.run.app/'
response = requests.get(ratemate_api_url, params=params)


your_personal_score = response.json()

if response.status_code == 200:
    # Retrieve prediction from JSON response
    score = response.json()["average score of ratings"]

    # Display prediction to the user
    st.success(f'Your personal score for the restaurant is: ⭐️ {your_personal_score} ⭐️')
else:
    st.error('Failed to get score from the API. Please try again.')

st.divider()

#st.header(f'Restuarant Rating: ⭐️ {original_score} ⭐️')
st.header(f'Your Personal Score: ⭐️ {your_personal_score} ⭐️')
