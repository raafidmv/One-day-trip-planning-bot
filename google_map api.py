import os
import streamlit as st
import folium
from streamlit_folium import folium_static
from dotenv import load_dotenv
import requests
import polyline

# Load environment variables (if using a .env file)
load_dotenv()
api_key = os.getenv("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_MAPS_API_KEY")

st.title("Kolkata Tour Route")

# Starting point: ITC Royal Bengal, Kolkata
start = (22.5449, 88.4033)  # Latitude, Longitude

# Notable tourist places in Kolkata (example list)
# Feel free to add/remove or reorder these places.
locations = [
    (22.5448, 88.3426),  # Victoria Memorial
    (22.5626, 88.3510),  # Indian Museum
    (22.5850, 88.3468),  # Howrah Bridge
    (22.5222, 88.3476),  # Kalighat Kali Temple
    (22.6557, 88.3539)   # Dakshineswar Kali Temple (End)
]

# The last location in this list will be our final destination
end = locations[-1]
# The waypoints are all locations except the last one
waypoints = locations[:-1]

# Convert waypoints to a string for the Directions API
waypoints_str = "|".join([f"{lat},{lng}" for lat, lng in waypoints])

# Build the Directions API request
directions_url = "https://maps.googleapis.com/maps/api/directions/json"
params = {
    "origin": f"{start[0]},{start[1]}",
    "destination": f"{end[0]},{end[1]}",
    "waypoints": f"optimize:true|{waypoints_str}",
    "key": api_key
}

response = requests.get(directions_url, params=params)
data = response.json()

if data.get("status") == "OK":
    route = data["routes"][0]
    overview_polyline = route["overview_polyline"]["points"]

    # Decode the polyline
    decoded_points = polyline.decode(overview_polyline)

    # Create a folium map centered on Kolkata
    # Approximate center of Kolkata
    m = folium.Map(location=[22.5726, 88.3639], zoom_start=12)

    # Add a marker for the start location
    folium.Marker(
        location=start,
        popup="Start (ITC Royal Bengal)",
        icon=folium.Icon(color='black', icon='info-sign')
    ).add_to(m)

    # Add a marker for the end location
    folium.Marker(
        location=end,
        popup="End (Dakshineswar Kali Temple)",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    # Mark the waypoints
    for i, wpt in enumerate(waypoints, start=1):
        folium.Marker(
            location=wpt,
            popup=f"Waypoint {i}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Draw the route on the map
    folium.PolyLine(decoded_points, color="blue", weight=5, opacity=0.8).add_to(m)

    # Calculate total distance and duration from route legs
    total_distance = 0
    total_duration = 0
    for leg in route["legs"]:
        total_distance += leg["distance"]["value"]  # in meters
        total_duration += leg["duration"]["value"]  # in seconds

    # Display total distance and duration
    st.write(f"**Total Distance:** {total_distance/1000:.2f} km")
    st.write(f"**Total Duration:** {total_duration/60:.2f} minutes")

    folium_static(m)
else:
    st.write("Failed to retrieve data from the Google Directions API. Please check your API key and request parameters.")
