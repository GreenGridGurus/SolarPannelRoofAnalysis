import streamlit as st
import os
import pandas as pd
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt



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

css_content = load_css()
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

st.title("Heatmap")

with st.container():

    # Create a 2-column layout
    col1, spacing, col2 = st.columns([1, 0.1, 1])  # Adjust the middle column's width for desired spacing

    # Left column content
    with col1:
        # Define coordinate systems
        from_crs = "EPSG:4326"  # WGS 84
        to_crs = "EPSG:31467"  # Gauss Krüger Zone 3

        # Create transformer object
        transformer = Transformer.from_crs(from_crs, to_crs)

        # Information extracted from the dataset header
        XLLCORNER = 3280500
        YLLCORNER = 5237500
        NROWS = 866
        NCOLS = 654
        CELLSIZE = 1000
        NODATA_VALUE = -999

        # Load data as 2d array
        data = np.loadtxt("radiationProject/grids_germany_annual_radiation_global_2022.asc", skiprows=28)
        data[data == -999] = np.nan

        for row in range(NROWS):
            for col in range(NCOLS):
                # Calculate the Gauss Krüger coordinates for the current cell
                h = XLLCORNER + (col * CELLSIZE) + (CELLSIZE / 2)
                r = YLLCORNER + ((NROWS - row - 1) * CELLSIZE) + (CELLSIZE / 2)

                # Convert Gauss Krüger coordinates to latitude and longitude
                latitude, longitude = transformer.transform(h, r)

                # Get the radiance value for the current cell
                radiance = data[row, col]

        # Get the indices of the 100 highest values in the flattened array
        indices = np.argsort(data.flatten())[-100:]

        # Convert the corresponding indices back to 2D coordinates
        y_coords, x_coords = np.unravel_index(indices, data.shape)

        # Convert the Gauss-Krüger coordinates back to latitude and longitude
        latitudes, longitudes = [], []
        for y, x in zip(y_coords, x_coords):
            # Calculate the Gauss-Krüger coordinates of the cell
            easting = XLLCORNER + x * CELLSIZE
            northing = YLLCORNER + (NROWS - y) * CELLSIZE

            # Convert the Gauss-Krüger coordinates to latitude and longitude
            latitude, longitude = transformer.transform(northing, easting, direction="INVERSE")
            latitudes.append(latitude)
            longitudes.append(longitude)

        # Create a Pandas DataFrame with the latitude and longitude of the 100 highest values
        df = pd.DataFrame({"latitude": latitudes, "longitude": longitudes})

        # Define the extent of the plot in longitude and latitude coordinates
        left = transformer.transform(YLLCORNER, XLLCORNER, direction="INVERSE")[1]
        right = transformer.transform(YLLCORNER, XLLCORNER + NROWS * CELLSIZE, direction="INVERSE")[1]
        bottom = transformer.transform(YLLCORNER + NROWS * CELLSIZE, XLLCORNER, direction="INVERSE")[0]
        top = transformer.transform(YLLCORNER, XLLCORNER, direction="INVERSE")[0]

        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create a heatmap of the data with longitude and latitude axes
        heatmap = ax.pcolormesh(np.arange(x - 0.5, x + data.shape[1] - 0.5),
                                np.arange(y - 0.5, y + data.shape[0] - 0.5)[::-1],
                                data, cmap="inferno")

        # Add a colorbar
        cbar = fig.colorbar(heatmap)

        # Set the title and axis labels
        ax.set_title("Annual Radiation Global in Germany (2022)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")


        # Show the plot
        st.pyplot(fig)



    # Right column content
    with col2:

        st.subheader("Most efficient Regions")
        location = "Sample Location"
        st.write(f"Location: {location}")
        file_path = os.path.join(os.path.dirname(__file__), "../citiesRadiation_NoDuplicates_Merged.csv")
        df = pd.read_csv(file_path)
        st.write(df)