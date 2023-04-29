from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="app")

village_name = "Hagen"
location = geolocator.geocode("München")
print(location.address)
print((location.latitude, location.longitude))


Latitude = "48.231886"
Longitude = "9.462613"
location = geolocator.reverse(Latitude + "," + Longitude)
# Display
address = location.raw['address']
print(address)
# traverse the data
city = address.get('village', '')

if city == village_name:
    print("True")
print(city)
