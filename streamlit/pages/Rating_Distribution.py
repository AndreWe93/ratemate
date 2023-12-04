import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import requests

from ml_logic.scrape_apify import scrape_apify
from ml_logic.NLP import new_column_NLP
from ml_logic.calculate_final_score import *
from interface.main import *
from params import *

st.markdown("""# RateMate
## Ratings Distribution""")

# Function to generate and display a word cloud
def generate_ratings_distribution(df):
    # Display Rating distribution
    fig = plt.figure(figsize=(10,10))
    sns.countplot(data=df, x="stars", palette="Spectral", hue="stars")
    st.pyplot(fig)


restaurant_url = st.text_input("url of your restaurant")

if st.button("Reviews"):
    #url = 'https://awtestratemate2-z2kqlvo2ta-ew.a.run.app/scrape'

    params = {
            'url': restaurant_url
            }
    url = restaurant_url


    # Make API request
    #response = requests.get(url, params=params)
    df = scrape_apify(url)
    df = df[COLUMN_NAMES_RAW]
    pre_processed_df = preprocess_reviews_text(df) # Still need to do the column selection

    # Streamlit app
    st.title("Distribution of ratings")

    # Display the word cloud
    generate_ratings_distribution(pre_processed_df)
