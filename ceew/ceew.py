import streamlit as st
import pandas as pd
import folium
import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import streamlit.components.v1 as components
import requests
import zipfile
import io
import os
csv_zip_url = 'https://github.com/MuhammedWaseemAli/delhioccurencehotspot-/blob/main/ceew/complaintcopiedcsv.zip?raw=true'
shapefile_zip_url = 'https://github.com/MuhammedWaseemAli/delhioccurencehotspot-/blob/main/ceew/delhi%20shape%20file.zip?raw=true'
def download_and_extract_zip(url, extract_to):
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(extract_to)
csv_extract_dir = 'extracted_csv'
shapefile_extract_dir = 'extracted_shapefile'
os.makedirs(csv_extract_dir, exist_ok=True)
os.makedirs(shapefile_extract_dir, exist_ok=True)
download_and_extract_zip(csv_zip_url, csv_extract_dir)
download_and_extract_zip(shapefile_zip_url, shapefile_extract_dir)
nested_shapefile_zip = os.path.join(shapefile_extract_dir, 'delhi shape file.zip')
def extract_nested_zip(zip_path, extract_to):
    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} to {extract_to}")
    else:
        print(f"File {zip_path} not found")
extract_nested_zip(nested_shapefile_zip, shapefile_extract_dir)
@st.cache_data
def load_data():
    csv_path = os.path.join(csv_extract_dir, 'complaintcopiedcsv.csv')
    if os.path.isfile(csv_path):
        complaints_df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        complaints_df = complaints_df.drop(complaints_df.index[1:57556])
        complaints_df[['Latitude', 'Longitude']] = complaints_df['Latitude & Longitude'].str.split(',', expand=True)
        complaints_df['Latitude'] = complaints_df['Latitude'].astype(float)
        complaints_df['Longitude'] = complaints_df['Longitude'].astype(float)
        return complaints_df
    else:
        raise FileNotFoundError(f"CSV file {csv_path} not found")
@st.cache_data
def load_shapefile():
    shapefile_dir = os.path.join(shapefile_extract_dir, 'delhi shape file')
    shapefile_path = os.path.join(shapefile_dir, 'Delhi_Wards.shp')
    if os.path.isfile(shapefile_path):
        wards_gdf = gpd.read_file(shapefile_path)
        if wards_gdf.crs is None:
            wards_gdf.set_crs(epsg=4326, inplace=True)
        else:
            wards_gdf.to_crs(epsg=4326, inplace=True)
        return wards_gdf
    else:
        raise FileNotFoundError(f"Shapefile {shapefile_path} not found")
def get_style_function():
    return lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1}
def create_map(offence_type, complaints_df, wards_gdf):
    wards_gdf = wards_gdf.copy()
    filtered_df = complaints_df[complaints_df['Offences'] == offence_type].copy()
    filtered_df['Latitude_rad'] = np.radians(filtered_df['Latitude'])
    filtered_df['Longitude_rad'] = np.radians(filtered_df['Longitude'])
    coords = filtered_df[['Latitude_rad', 'Longitude_rad']].values
    db = DBSCAN(eps=0.00000085, min_samples=1, metric='haversine').fit(coords)
    filtered_df['Cluster'] = db.labels_
    clustered_counts = filtered_df.groupby('Cluster').size().reset_index(name='Occurrences')
    cluster_centers = filtered_df.groupby('Cluster').agg({
        'Latitude': 'mean',
        'Longitude': 'mean',
        'Geo Location': 'first',
    }).reset_index()
    clustered_data = pd.merge(cluster_centers, clustered_counts, on='Cluster')
    top_100_occurrences = clustered_data.sort_values(by='Occurrences', ascending=False).head(100)
    norm = mcolors.Normalize(vmin=top_100_occurrences['Occurrences'].min(), vmax=top_100_occurrences['Occurrences'].max())
    cmap = cm.get_cmap('YlOrRd')
    m = folium.Map(location=[28.7041, 77.1025], zoom_start=11)
    folium.GeoJson(
        wards_gdf,
        name='Delhi Wards',
        style_function=get_style_function()
    ).add_to(m)
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
            radius=5 + row['Occurrences'] / 10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300, max_height=600)
        ).add_to(m)
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
st.set_page_config(page_title="Delhi Complaints Map", layout="wide")


hide_streamlit_style = """
    <style>
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    .stApp { background: white; }
    .viewerBadge_link__1S137 { display: none !important; }
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom page styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
        background: url("https://images.pexels.com/photos/15057524/pexels-photo-15057524/free-photo-of-group-of-cheerful-friends.jpeg");
        background-size: cover;
    }
    .title {
        font-size: 36px;
        color: white;
        text-align: center;
        padding: 10px;
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 5px;
    }
    .description {
        font-size: 18px;
        color: white;
        text-align: center;
        padding: 10px;
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Delhi Complaints Map</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="description">Explore the top 100 complaint hotspots in Delhi based on different offence types.click on the spots on map to get images and status details.</div>',
    unsafe_allow_html=True)

# Load data
complaints_df = load_data()
wards_gdf = load_shapefile()

offence_types = complaints_df['Offences'].unique()
selected_offence_type = st.selectbox("Select an offence type to visualize:", offence_types)

m, top_100_occurrences = create_map(selected_offence_type, complaints_df, wards_gdf)
map_html = f"{m.get_root().render()}"

components.html(map_html, height=700)

st.dataframe(top_100_occurrences[['Occurrences', 'Geo Location', 'Latitude', 'Longitude']], height=300)

