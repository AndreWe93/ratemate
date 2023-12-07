import requests
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud





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

url = st.text_input("url of your restaurant")

# the selected value is returned by st.slider
price_review_weightage = st.slider('Select your price rating weight', 0.0, 1.0, 0.25)
food_review_weightage = st.slider('Select your food rating weight', 0.0, 1.0, 0.25)
service_review_weightage = st.slider('Select your service rating weight', 0.0, 1.0, 0.25)
ambience_review_weightage = st.slider('Select your ambience rating weight', 0.0, 1.0, 0.25)


if st.button("Get Score"):

    params = {
            'url': url,
            'price_review_weightage': price_review_weightage,
            'food_review_weightage': food_review_weightage,
            'service_review_weightage': service_review_weightage,
            'ambience_review_weightage': ambience_review_weightage
            }


    ratemate_api_url = 'https://ratemate-z2kqlvo2ta-ew.a.run.app/personal_score'
    response = requests.get(ratemate_api_url, params=params)

    # original_score = response.json()
    your_personal_score = response.json()["personal_score"]
    top_1_review = response.json()["top_1"]
    top_2_review = response.json()["top_2"]
    top_3_review = response.json()["top_3"]
    sub_price = response.json()["sub_price"]
    sub_service = response.json()["sub_service"]
    sub_atmosphere = response.json()["sub_atmosphere"]
    sub_food = response.json()["sub_food"]
    wordcloud_input = response.json()["wordcloud_input"]
    dist_price = response.json()["dist_price"]
    dist_service = response.json()["dist_service"]
    dist_atmosphere = response.json()["dist_atmosphere"]
    dist_food = response.json()["dist_food"]

    st.divider()

    #st.header(f'Restuarant Rating: ⭐️ {original_score} ⭐️')
    st.header(f'⭐️ Your personal score is: {your_personal_score} ⭐️')
    st.header("Here are the reviews of the most active reviewers")
    st.markdown(f'Top 1: {top_1_review}')
    st.markdown(f'Top 2: {top_2_review}')
    st.markdown(f'Top 3: {top_3_review}')
    st.header("Here are the sub ratings")
    st.markdown(f'sub rating for price: {sub_price}')
    st.markdown(f'sub rating for service: {sub_service}')
    st.markdown(f'sub rating for atmosphere: {sub_atmosphere}')
    st.markdown(f'sub rating for food: {sub_food}')
    st.header("Here are the topic distributions")
    st.markdown(f'distribution for price: {dist_price}')
    st.markdown(f'distribution for service: {dist_service}')
    st.markdown(f'distribution for atmosphere: {dist_atmosphere}')
    st.markdown(f'distribution for food: {dist_food}')

    wordcloud = WordCloud(max_words=10000, min_font_size=10, height=800, width=1600,
               background_color="white", colormap="viridis").generate(wordcloud_input)

    # Display the word cloud using Matplotlib
    fig = plt.figure(figsize=(20,20))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)
