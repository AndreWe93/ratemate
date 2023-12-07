import streamlit as st
import requests
import gmaps
import googlemaps
import pydeck as pdk

api_key = "AIzaSyBH2zXte15didv6k_rGf4dOqx4iw4scS8k"

# ####### TO Dos #########
# """
# 1. Show the original score
# 2. Show the subratings from the dataset
# 3. Integrate the Word cloud picture from my notebook somehow into the streamlit app
# 4. Add more features
# """
st.set_page_config(layout="wide")
st.sidebar.markdown("""<style> .sidebar .sidebar-content { width: 400px; } </style>""", unsafe_allow_html=True)

st.sidebar.title("About")
st.sidebar.info(
    """
    This app is helping you to predict personal rating.
    """)


# gmaps.configure(api_key=api_key)

# new_york_coordinates = (40.75, -74.00)

# fig = gmaps.figure(center=new_york_coordinates, zoom_level=12)

# st.sidebar.write(fig)

# def show_google_maps(api_key, lat, lon):
#     #gmaps = googlemaps.Client(key=api_key)
#     gmaps.configure(api_key=api_key)

#     coordinates = (lat,lon)

#     fig = gmaps.figure(center=coordinates, zoom_level=12)

#     return st.sidebar.write(fig)

def show_google_maps(lat, lon):
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12)
    map_ = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        api_keys={"mapbox": "YOUR_MAPBOX_API_KEY"}
    )
    st.sidebar.pydeck_chart(map_)


st.snow()

st.markdown("""# RateMate
## Get your personal rating for the restaurant of your choice
Hello friend, please enter the name of the restaurant you are interested in:""")



gmaps = googlemaps.Client(key=api_key)
@st.cache_data(ttl=500)
def find_restaurant(place_name):
    places = gmaps.places(query=place_name, type='restaurant')

    if places['results']:
        restaurant = places['results'][0]
        return restaurant
    else:
        return None



@st.cache_data(ttl=300)
def results_for_restorant(restaurant_name, search_button):
    results = []
    url = None
    lat = None
    lon = None

    if search_button:
        places = gmaps.places(query=restaurant_name, type='restaurant')
        #st.write(restaurant_name)
        restaurant = places['results'][0]
        if restaurant:
            results.append(f" {restaurant['name']}")
            results.append(f"Address: {restaurant['formatted_address']}")
            results.append(f"Rating: {restaurant.get('rating', 'No rating')}")
            place_id = restaurant['place_id']
            google_maps_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
            url = google_maps_url
            lat = restaurant['geometry']['location']['lat']
            lon = restaurant['geometry']['location']['lng']
        else:
            results.append("No restaurant found")
    return url, results, lat, lon

restaurant_name = st.text_input("Enter restaurant name and press Enter", "Type here")
search_button = st.button("Find")



url, results, lat, lon = results_for_restorant(restaurant_name, search_button)
show_google_maps(lat, lon)
for result in results:
    st.markdown(f"<p style='color: #000080;'>{result}</p>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='color: grey;'>❗️ If this is not the restaurant you are looking for, please specify the search string, e.g. by entering a street.</h6>",
    unsafe_allow_html=True
)


# empty = st.container()
# empty.write(url)
# empty.write()


# empty2 = st.container()
dash1= st.container()

local_guides_review_weightage = st.checkbox('Review only from local guides')

with dash1:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("<h4 style='text-align: center; color: #800080;'>Specify your preferences</h4>", unsafe_allow_html=True)

    with col5:
        price_review_weightage = st.slider('PRICE', 0.0, 1.0, 0.25)

    with col2:
        food_review_weightage = st.slider('FOOD', 0.0, 1.0, 0.25)

    with col3:
        service_review_weightage = st.slider('SERVICE ', 0.0, 1.0, 0.25)

    with col4:
        ambience_review_weightage = st.slider('AMBIENCE', 0.0, 1.0, 0.25)


search_button2 = st.button("Get Score")


# the selected value is returned by st.slider
# checkbox for local guides
if search_button2:
    st.markdown('✅ Great, this restaurant has been chosen:')
    st.session_state.is_enter_pressed = False

    restaurant_name = find_restaurant(restaurant_name)
    url, results, lat, lon = results_for_restorant(restaurant_name['name'], search_button2)
    for result in results:
        st.write(result)
    with st.spinner('😎 Please wait for it...'):

        st.markdown("<h3 style='text-align: center;'>PREDICTING</h3>", unsafe_allow_html=True)

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
        #st.write(url)
        if response.status_code == 200:
            your_personal_score = response.json()
            st.success("Prediction Complete!")
            st.header(f'⭐️ {your_personal_score} ⭐️')
        else:
            st.error("Error in prediction. Please try again.")

    # for result in results:
    #     st.write(result, url)
