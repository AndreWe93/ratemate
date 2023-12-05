import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import streamlit as st
import requests

from ml_logic.scrape_apify import scrape_apify
from interface.main import *
from params import *

st.markdown("""# RateMate
## Reviews Overview""")

# '''
# # TODO:
# 1. Show the top three reviews from the top three reviewers
# 2. Top three reviewers are reviewers who wrote the most number of reviews
# '''

# Function to generate and display a word cloud
def generate_wordcloud(df):
    wordcloud = WordCloud(max_words=10000, min_font_size=10, height=800, width=1600,
               background_color="white", colormap="viridis").generate(" ".join(df["reviews_without_SW"].astype(str)))

    # Display the word cloud using Matplotlib
    fig = plt.figure(figsize=(20,20))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)

restaurant_url = st.text_input("url of your restaurant")

if st.button("Reviews"):

    params = {
            'url': restaurant_url
            }
    url = restaurant_url


    # Make API request
    #response = requests.get(url, params=params)
    # df = scrape_apify(url)

    # Placeholder code
    df = pd.read_csv("./raw_data_slim/merged_slim_file.csv")

    df = df[COLUMN_NAMES_RAW]
    pre_processed_df = preprocess_reviews_text(df) # Still need to do the column selection

    # Streamlit app
    st.title("Reviews Context")

    # Display the word cloud
    generate_wordcloud(pre_processed_df)
