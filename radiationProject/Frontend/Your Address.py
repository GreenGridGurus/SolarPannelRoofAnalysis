import googlemaps
import streamlit as st
import os
import gmaps
import config

st.set_page_config(
    page_title="Heatmap",
    page_icon="👋",
    layout="wide"
)

def load_assets():
    # Load CSS
    css_file_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_file_path, "r") as f:
        css = f.read()

    # Load JavaScript
    js_file_path = os.path.join(os.path.dirname(__file__), "fixes.js")
    with open(js_file_path, "r") as f:
        js = f.read()
    return css, js


css_content, js_content = load_assets()
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
st.markdown(f"<script>{js_content}</script>", unsafe_allow_html=True)

with st.container():
    st.title("Your data")
    # Create a 2-column layout
    col1, spacing, col2 = st.columns([1, 0.1, 1])  # Adjust the middle column's width for desired spacing

    # Left column content
    with col1:
        # Set the coordinates for the map
        latitude = 37.57749
        longitude = -122.4194

        # Initialize the Google Maps client
        gmaps = googlemaps.Client(key=config.API_KEY)

        # Use streamlit.text_input() to get the user's address input
        address = st.text_input("Enter your address:")

        if address:
            geocode_result = gmaps.geocode(address)

            if geocode_result:
                latitude = geocode_result[0]["geometry"]["location"]["lat"]
                longitude = geocode_result[0]["geometry"]["location"]["lng"]



        # Define the HTML string that will contain the map
        html_string = f"""
        <!DOCTYPE html>
        <div class="map-container">
            <iframe class="map-frame" width="100%" height="500px"  src="https://maps.google.com/maps?q={latitude},{longitude}&t=k&output=embed"></iframe>
        </div>
        """
        # Display the HTML iframe using the st.components.v1.html function
        # st.components.v1.html(html_string)
        st.components.v1.html(html_string, height=500)


    # Right column content
    with col2:
        st.subheader("Information about your location")
        location = "Sample Location"
        roof_size = 100
        radiation = 200
        max_pv_output = 300
        max_panels = 10
        max_energy = 400
        euro_generated = 500
        efficiency = 600
        azimuth = 700

        # Create two columns inside col2
        info_col, value_col = st.columns([0.5, 1])

        with info_col:
            st.markdown("Location:")
            st.markdown("Size of roof:")
            st.markdown("Radiation:")
            st.markdown("Max PV output:")
            st.markdown("Max panels on roof:")
            st.markdown("Max energy:")
            st.markdown("Efficiency:")
            st.markdown("Azimuth:")
            st.markdown("€ generated:")

        with value_col:
            st.markdown(f"**{location}**")
            st.markdown(f"**{roof_size}**")
            st.markdown(f"**{radiation}**")
            st.markdown(f"**{max_pv_output}**")
            st.markdown(f"**{max_panels}**")
            st.markdown(f"**{max_energy}**")
            st.markdown(f"**{efficiency}**")
            st.markdown(f"**{azimuth}**")
            st.markdown(f"**{euro_generated}**")

    col1, col2, col3 = st.columns(3)


    file_path = os.path.join(os.path.dirname(__file__), "image.png")
    col1.image(file_path, caption="Image 1", use_column_width=True)
    col2.image(file_path, caption="Image 2", use_column_width=True)
    col3.image(file_path, caption="Image 3", use_column_width=True)