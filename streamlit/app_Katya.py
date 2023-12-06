import streamlit as st
import requests
import time
import gmaps
import googlemaps

api_key = "AIzaSyBH2zXte15didv6k_rGf4dOqx4iw4scS8k"

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



gmaps = googlemaps.Client(key=api_key)

def find_restaurant(place_name):
    places = gmaps.places(query=place_name, type='restaurant')

    if places['results']:
        restaurant = places['results'][0]
        return restaurant
    else:
        return None


restaurant_name = st.text_input("Enter restaurant name and press Enter", "Type here")
search_button = st.button("Find")
results = []

if search_button:
    restaurant = find_restaurant(restaurant_name)
    if restaurant:
        results.append(f"Found the restaurant: {restaurant['name']}")
        results.append(f"Address: {restaurant['formatted_address']}")
        results.append(f"Rating: {restaurant.get('rating', 'No rating')}")
        place_id = restaurant['place_id']
        google_maps_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        url = google_maps_url
    else:
        results.append("No restaurant found")

for result in results:
    st.write(result)

empty = st.container()


# empty2 = st.container()
dash1= st.container()

local_guides_review_weightage = st.checkbox('Review only from local guides')

with dash1:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("""##Specify your preferences""")

    with col5:
        price_review_weightage = st.slider('PRICE', 0.0, 1.0, 0.25)

    with col2:
        food_review_weightage = st.slider('FOOD', 0.0, 1.0, 0.25)

    with col3:
        service_review_weightage = st.slider('SERVICE ', 0.0, 1.0, 0.25)

    with col4:
        ambience_review_weightage = st.slider('AMBIENCE', 0.0, 1.0, 0.25)


# the selected value is returned by st.slider
# checkbox for local guides
if st.button("Get Score"):
    st.session_state.is_enter_pressed = False

    st.markdown("<h2 style='text-align: center;'>PREDICTING</h2>", unsafe_allow_html=True)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    for i in range(101):
        time.sleep(1)
        progress_bar.progress(i)
        progress_text.text(f"Progress: {i}%")

    params = {
        'url': url,
        'price_review_weightage': price_review_weightage,
        'food_review_weightage': food_review_weightage,
        'service_review_weightage': service_review_weightage,
        'ambience_review_weightage': ambience_review_weightage,
        'local_guides_review_weightage': local_guides_review_weightage
    }

    ratemate_api_url = 'https://ratemate-z2kqlvo2ta-ew.a.run.app/personal_score'
    response = requests.get(ratemate_api_url, params=params)

    if response.status_code == 200:
        your_personal_score = response.json()
        st.success("Prediction Complete!")
        st.header(f'⭐️ {your_personal_score} ⭐️')
    else:
        st.error("Error in prediction. Please try again.")



# gmaps.configure(api_key=api_key)

# new_york_coordinates = (40.75, -74.00)

# fig = gmaps.figure(center=new_york_coordinates, zoom_level=12)

# st.write(fig)
