import math
import pandas as pd
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
from pyrosm import OSM
from geopy.geocoders import Nominatim
import geopolygon as gp
import osmnx as ox
from shapely.geometry import Point, Polygon
import geopandas as gpd
from pyrosm import OSM

import requests
import cv2

# Coordinates of TU Munich
latitude, longitude = 48.1496636, 11.5656715

# Define coordinate systems
from_crs = "EPSG:4326"  # WGS 84
to_crs = "EPSG:31467"  # Gauss Krüger Zone 3

# Create transformer object
transformer = Transformer.from_crs(from_crs, to_crs)

# Convert latitude and longitude to Gauss Krüger coordinates
h, r = transformer.transform(latitude, longitude)
# Information extracted from the dataset header
XLLCORNER = 3280500
YLLCORNER = 5237500
NROWS = 866
NCOLS = 654
CELLSIZE = 1000
NODATA_VALUE = -999

# Load data as 2d array
data = np.loadtxt("grids_germany_annual_radiation_global_2022.asc", skiprows=28)
data[data == -999] = np.nan

y, x = math.floor((r - XLLCORNER) / CELLSIZE), NROWS - math.ceil((h - YLLCORNER) / CELLSIZE)

radiance = data[x, y]

for row in range(NROWS):
    for col in range(NCOLS):
        # Calculate the Gauss Krüger coordinates for the current cell
        h = XLLCORNER + (col * CELLSIZE) + (CELLSIZE / 2)
        r = YLLCORNER + ((NROWS - row - 1) * CELLSIZE) + (CELLSIZE / 2)

        # Convert Gauss Krüger coordinates to latitude and longitude
        latitude, longitude = transformer.transform(h, r)

        # Get the radiance value for the current cell
        radiance = data[row, col]

        # Print the latitude, longitude, and radiance value for the current cell
        #print(f"Latitude: {latitude}, Longitude: {longitude}, Radiance: {radiance}")

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

# Print the first 5 rows of the DataFrame
#print(df)

# Define the extent of the plot in longitude and latitude coordinates
left = transformer.transform(YLLCORNER, XLLCORNER, direction="INVERSE")[1]
right = transformer.transform(YLLCORNER, XLLCORNER + NROWS * CELLSIZE, direction="INVERSE")[1]
bottom = transformer.transform(YLLCORNER + NROWS * CELLSIZE, XLLCORNER, direction="INVERSE")[0]
top = transformer.transform(YLLCORNER, XLLCORNER, direction="INVERSE")[0]

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 10))

# Create a heatmap of the data with longitude and latitude axes
heatmap = ax.pcolormesh(np.arange(x - 0.5, x + data.shape[1] - 0.5), np.arange(y - 0.5, y + data.shape[0] - 0.5)[::-1],
                        data, cmap="inferno")

# Add a colorbar
cbar = fig.colorbar(heatmap)

# Set the title and axis labels
ax.set_title("Annual Radiation Global in Germany (2022)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Show the plot
plt.show()
village_name = "Herdwangen-Schönach"
geolocator = Nominatim(user_agent="app_make")
location = geolocator.geocode(village_name)

village_coords = (location.latitude, location.longitude)


print(f"Radiance at TU Munich: {radiance} kWh/m²")



# Get place boundary related to the place name as a geodataframe
poly = ox.geocode_to_gdf(village_name)
print(f'area: {poly}')


# assume these are the latitude and longitude coordinates to check
lat = 50.2
lon = 6.3

# create a point object from the latitude and longitude coordinates
point = Point(lon, lat)
print(point.within(poly))
if point.within(poly):
    print("The point is inside the village.")
else:
    print("The point is outside the village.")



osm = OSM('buildings/luxembourg-latest.osm.pbf')
buildings = osm.get_buildings()
pd.set_option('display.max_rows', None)

print(f'Number of buildings: {len(buildings)}')
#print(buildings.head()['geometry'].centroid)
centroids = buildings['geometry'].centroid
print(f'centroids: {centroids}')
coords = []
for point in centroids:
    for coord in point.coords[0]:
        if int(coord.x) == int(village_coords[0]) and int(coord.y) == int(village_coords[1]):
            coords.append(coord)

#xy_arrays = [[coord for coord in point.coords[0]] for point in centroids]

# print the list of x-y arrays\
print(f'coords: {coords}')
for coord in coords:
    Longitude = coord[0]
    Latitude = coord[1]

    print(f'Latitude: {Latitude}, Longitude: {Longitude}')

    location = geolocator.reverse(str(Latitude) + "," + str(Longitude))
    print(f'location: {location}')
    address = location.raw['address']
    # traverse the data
    city = address.get('village', '')

    if city == village_name:
        print(address)

#print(xy_arrays)

#buildings.to_csv('data.csv', index=False)
#pd.set_option("display.max_rows", 100)


# url = "https://maps.googleapis.com/maps/api/staticmap?center=40.714728,-73.998672&zoom=12&maptype=satellite&size=400x400&key=AIzaSyA3kg7YWugGl1lTXmAmaBGPNhDW9pEh5bo&signature=5tyWj9NAOGlFz33nroLk6sV4ASk="
# response = requests.get(url).content
# image = cv2.imdecode(np.frombuffer(response, np.uint8), cv2.IMREAD_UNCHANGED)
#
# cv2.imshow("Satellite Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# osm = OSM('bremen-latest.osm.pbf')
# buildings = osm.get_buildings()


def avgRadiation(city_name):
    cities = pd.read_csv("cities")
    city_rows = cities[cities['city'] == city_name]
    if len(city_rows) == 0:
        raise ValueError(f"No data found for city '{city_name}'.")
    avg_radiation = city_rows['radiation'].mean()
    return avg_radiation
