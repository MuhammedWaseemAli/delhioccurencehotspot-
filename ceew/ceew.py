pip install streamlit pandas folium geopandas scikit-learn numpy matplotlib
pip freeze > requirements.txt


import streamlit as st
import pandas as pd
import folium
import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import streamlit.components.v1 as components

# Load the CSV file with a specified encoding
csv_path = r'C:\Users\wasee\OneDrive\Desktop\ceew project\complaints data analysis\complaintcopiedcsv.csv'
complaints_df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Remove rows 2 to 57556
complaints_df = complaints_df.drop(complaints_df.index[1:57556])

# Split latitude and longitude into separate columns
complaints_df[['Latitude', 'Longitude']] = complaints_df['Latitude & Longitude'].str.split(',', expand=True)
complaints_df['Latitude'] = complaints_df['Latitude'].astype(float)
complaints_df['Longitude'] = complaints_df['Longitude'].astype(float)

# Load the shapefile
shapefile_path = r'C:\Users\wasee\Downloads\Delhi_Wards.shp'
wards_gdf = gpd.read_file(shapefile_path)

# Set CRS to EPSG:4326 if not already set
if wards_gdf.crs is None:
    wards_gdf.set_crs(epsg=4326, inplace=True)
else:
    wards_gdf.to_crs(epsg=4326, inplace=True)

# Define a function to create a map based on offence type
def create_map(offence_type):
    # Filter for specific offences
    filtered_df = complaints_df[complaints_df['Offences'] == offence_type].copy()

    # Convert coordinates to radians for haversine calculation
    filtered_df['Latitude_rad'] = np.radians(filtered_df['Latitude'])
    filtered_df['Longitude_rad'] = np.radians(filtered_df['Longitude'])

    # Apply DBSCAN with a 5-meter radius (approximately 0.000045 radians) and minimum samples of 1
    coords = filtered_df[['Latitude_rad', 'Longitude_rad']].values
    db = DBSCAN(eps=0.00000085, min_samples=1, metric='haversine').fit(coords)

    # Add cluster labels to the DataFrame
    filtered_df['Cluster'] = db.labels_

    # Group by clusters and count occurrences
    clustered_counts = filtered_df.groupby('Cluster').size().reset_index(name='Occurrences')

    # Get the mean location for each cluster
    cluster_centers = filtered_df.groupby('Cluster').agg({
        'Latitude': 'mean',
        'Longitude': 'mean',
        'Geo Location': 'first',  # Take the first occurrence of Geo Location as representative
    }).reset_index()

    # Merge the counts with the cluster centers
    clustered_data = pd.merge(cluster_centers, clustered_counts, on='Cluster')

    # Tabulate top 100 occurrences
    top_100_occurrences = clustered_data.sort_values(by='Occurrences', ascending=False).head(100)

    # Normalize occurrences for colormap
    norm = mcolors.Normalize(vmin=top_100_occurrences['Occurrences'].min(), vmax=top_100_occurrences['Occurrences'].max())
    cmap = cm.get_cmap('YlOrRd')

    # Create a folium map centered around Delhi
    m = folium.Map(location=[28.7041, 77.1025], zoom_start=11)

    # Add ward boundaries to the map
    folium.GeoJson(
        wards_gdf,
        name='Delhi Wards',
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1}
    ).add_to(m)

    # Add circle markers for each of the top 100 clusters with occurrences count
    for idx, row in top_100_occurrences.iterrows():
        cluster_complaints = filtered_df[filtered_df['Cluster'] == row['Cluster']]
        resolve_images = ''.join([f"<img src='{img}' width='150' height='150'><br>" for img in cluster_complaints['Resolve Image'].dropna()])
        offence_images = ''.join([f"<img src='{img}' width='150' height='150'><br>" for img in cluster_complaints['Offence Image'].dropna()])
        color = mcolors.to_hex(cmap(norm(row['Occurrences'])))
        popup_content = (f"Occurrences: {row['Occurrences']}<br>"
                         f"Location: {row['Geo Location']}<br>"
                         f"Date: {', '.join(cluster_complaints['Date and Time'].dropna())}<br>"
                         f"Status: {', '.join(cluster_complaints['Status'].dropna())}<br>"
                         f"Resolve Images:<br>{resolve_images}<br>"
                         f"Offence Images:<br>{offence_images}")
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5 + row['Occurrences'] / 10,  # Increase radius based on occurrences
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300, max_height=600)
        ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 250px; height: 150px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white;
        padding: 10px;
        ">
        <b>Occurrences Legend</b><br>
        <i style="background:#ffffb2; width: 20px; height: 20px; display: inline-block;"></i>&nbsp;Low (1-3)<br>
        <i style="background:#fd8d3c; width: 20px; height: 20px; display: inline-block;"></i>&nbsp;Medium (4-10)<br>
        <i style="background:#bd0026; width: 20px; height: 20px; display: inline-block;"></i>&nbsp;High (>10)<br>
    </div>
    '''

    m.get_root().html.add_child(folium.Element(legend_html))

    return m, top_100_occurrences

# Streamlit app setup
st.set_page_config(page_title="Delhi Complaints Map", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
        background: url("https://example.com/background.jpg");
        background-size: cover;
    }
    .title {
        font-size: 36px;
        color: white;
        text-align: center;
        padding: 10px;
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        margin-top: 20px;
    }
    .description {
        font-size: 18px;
        color: white;
        text-align: center;
        padding: 5px;
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
    }
    .dataframe {
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.8);
        margin: 0 auto;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Delhi top 100 Complaints hotspots by Offence Type</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Just select an offence from below you will see top 100 spots where more complaints are coming and also its pictures of complaint and statuses.</div>', unsafe_allow_html=True)

# Define offence types
offence_types = [
    'Dumping of Construction & Demolition Waste',
    'Air Pollution due to industries',
    'Illegal dumping of Garbage on road sides/ vacant land',
    'Burning of Biomass/garden waste',
    'Burning of garbage/plastic waste',
    'Air pollution from the sources other than Industry',
    'Potholes on Roads',
    'Road Dust',
    'Dust Pollution due to Construction/ Demolition activity',
    'Sale and Storage of banned SUP items',
    'Mfg. of banned SUP items in non Industrial Area',
    'Noise pollution from the sources other than Industry',
    'Visible smoke from vehicle exhaust'   
]

# Dropdown for offence selection
selected_offence = st.selectbox('Select Offence Type:', offence_types)

# Create and display map and tabulation for the selected offence
m, top_100_occurrences = create_map(selected_offence)

# Save map to HTML
map_html = m._repr_html_()

# Display the map in Streamlit using components.html
components.html(map_html, height=600)

# Display the top 10 locations and their occurrence numbers
top_10_locations = top_100_occurrences[['Occurrences', 'Geo Location']].head(10)
st.markdown('<div class="dataframe">', unsafe_allow_html=True)
st.write("Top 10 Locations with Occurrences:")
st.dataframe(top_10_locations)
