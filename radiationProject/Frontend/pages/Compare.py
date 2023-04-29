import googlemaps
import streamlit as st
import os
import gmaps
import pandas as pd
import configparser


st.set_page_config(
    page_title="Heatmap",
    page_icon="👋",
    layout="wide"
)
def load_css():
    file_path = os.path.join(os.path.dirname(__file__), "../styles.css")
    with open(file_path, "r") as f:
        css = f.read()
    return css

css_file = "styles.css"
css_content = load_css()
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


with st.container():
    st.title("Your data")
    # Create a 2-column layout
    col1, spacing, col2 = st.columns([1, 0.1, 1])  # Adjust the middle column's width for desired spacing
    latitude = 37.57749
    longitude = -122.4194

    file_path = os.path.join(os.path.dirname(__file__), "../cities_with_radiation.csv")
    df = pd.read_csv(file_path)

    # define the target latitude and longitude
    target_lat = 40.7128
    target_long = -74.0060

    # Left column content
    # Initialize the Google Maps client
    config = configparser.ConfigParser()
    file_path = os.path.join(os.path.dirname(__file__), "../config.ini")
    config.read(file_path)
    api_key = config['googleAPI']['API_KEY']

    gmaps = googlemaps.Client(key=api_key)
    with col1:
        # Use streamlit.text_input() to get the user's address input
        address = st.text_input("Enter your address:")

        if address:
            geocode_result = gmaps.geocode(address)

            if geocode_result:
                latitude = geocode_result[0]["geometry"]["location"]["lat"]
                longitude = geocode_result[0]["geometry"]["location"]["lng"]

                # write the latitude and longitude to the screen
                st.write(f"Latitude: {latitude}")
                st.write(f"Longitude: {longitude}")

                # calculate the distance between each row's latitude/longitude and the target location
                df['distance'] = ((df['latitude'] - latitude) ** 2 + (df['longitude'] - longitude) ** 2) ** 0.5

                # sort the dataframe by distance in ascending order
                df = df.sort_values('distance')

                # select the city from the row with the smallest distance
                best_city = df.iloc[0]['city']

                best_city_info = df.iloc[0][['radiation', 'city', 'state', 'extra', 'county', 'latitude', 'longitude']]

                # create a new dataframe with the desired information
                best_city_df = pd.DataFrame({'Information': best_city_info.index, 'Value': best_city_info.values})

                # display the dataframe as a grid using Streamlit
                st.write('Best city information:')
                st.write(best_city_df)

    # Right column content
    with col2:
        # Use streamlit.text_input() to get the user's address input
        address2 = st.text_input("Enter your address: ")

        if address2:
            geocode_result2 = gmaps.geocode(address2)

            if geocode_result2:
                latitude = geocode_result2[0]["geometry"]["location"]["lat"]
                longitude = geocode_result2[0]["geometry"]["location"]["lng"]

                # write the latitude and longitude to the screen
                st.write(f"Latitude: {latitude}")
                st.write(f"Longitude: {longitude}")

                # calculate the distance between each row's latitude/longitude and the target location
                df['distance'] = ((df['latitude'] - latitude) ** 2 + (df['longitude'] - longitude) ** 2) ** 0.5

                # sort the dataframe by distance in ascending order
                df = df.sort_values('distance')

                # select the city from the row with the smallest distance
                best_city = df.iloc[0]['city']

                best_city_info = df.iloc[0][['radiation', 'city', 'state', 'extra', 'county', 'latitude', 'longitude']]

                # create a new dataframe with the desired information
                best_city_df = pd.DataFrame({'Information': best_city_info.index, 'Value': best_city_info.values})

                # display the dataframe as a grid using Streamlit
                st.write('Best city information:')
                st.write(best_city_df)