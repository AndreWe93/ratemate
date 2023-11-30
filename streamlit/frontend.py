# pip install streamlit
import streamlit as st
import requests


'''
# Let's test our API with some scraping and scoring
'''

st.markdown('''
Hello friend, please insert the google maps url of the restaurant you are interested in:
''')


url_user = st.text_input("url of your restaurant")



if st.button("Get Score"):
    url = 'https://awtestratemate2-z2kqlvo2ta-ew.a.run.app/scrape'

    params = {
            'url': url_user
            }

    # Make API request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        # Retrieve prediction from JSON response
        score = response.json()["average score of ratings"]

        # Display prediction to the user
        st.success(f'The score of your restaurant is: {score}')
    else:
        st.error('Failed to get score from the API. Please try again.')
