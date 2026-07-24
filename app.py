import streamlit as st
import folium
from folium.plugins import Geocoder
from streamlit_folium import st_folium
import requests
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="NSRDB Downloader", layout="centered")

# Output column order/names, matching the reference export layout
OUTPUT_COLUMNS = ["Day", "Time", "GHI", "DNI", "DIF", "TEMP", "WS", "WD", "RH", "AP", "PWAT"]


def _find_column(df, *aliases):
    """Look up an NSRDB response column by its known possible names."""
    for name in aliases:
        if name in df.columns:
            return df[name]
    raise KeyError(f"None of {aliases} found in NSRDB response columns: {list(df.columns)}")


def parse_nsrdb_response(raw_text):
    """Parse the raw NSRDB CSV response into a DataFrame (skipping the two metadata rows)."""
    return pd.read_csv(io.StringIO(raw_text), skiprows=2)


def build_output_csv(df):
    """Convert a parsed NSRDB DataFrame into a CSV with just the column names and data rows."""
    timestamps = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]]).dt.tz_localize("UTC")

    out = pd.DataFrame({
        "Day": timestamps.dt.dayofyear,
        "Time": timestamps.dt.strftime("%H:%M"),
        "GHI": _find_column(df, "GHI"),
        "DNI": _find_column(df, "DNI"),
        "DIF": _find_column(df, "DHI"),
        "TEMP": _find_column(df, "Temperature", "Air Temperature"),
        "WS": _find_column(df, "Wind Speed"),
        "WD": _find_column(df, "Wind Direction"),
        "RH": _find_column(df, "Relative Humidity"),
        "AP": _find_column(df, "Pressure", "Surface Pressure"),
        "PWAT": _find_column(df, "Precipitable Water", "Total Precipitable Water"),
    })[OUTPUT_COLUMNS]

    return out.to_csv(index=False, lineterminator="\n"), out


def check_ghi_closure(df):
    """Compare cumulative GHI against DHI + DNI * cos(solar zenith angle) as a data-quality check."""
    zenith_rad = np.radians(_find_column(df, "Solar Zenith Angle"))
    calculated_ghi = _find_column(df, "DHI") + _find_column(df, "DNI") * np.cos(zenith_rad)

    ghi_sum = _find_column(df, "GHI").sum()
    calculated_sum = calculated_ghi.sum()
    diff_pct = abs(ghi_sum - calculated_sum) / ghi_sum * 100 if ghi_sum else 0

    return ghi_sum, calculated_sum, diff_pct

st.title("☀️ NSRDB Meteorological Data Downloader")
st.write("Download solar irradiance and weather data directly from the National Solar Radiation Database.")

# --- Authentication Section ---
st.markdown("### 1. Authentication")
col1, col2 = st.columns([3, 1], gap="medium")
with col1:
    api_key = st.text_input("NSRDB API Key", type="password", help="Enter your developer key")
with col2:
    # Adding some vertical space so the link aligns with the text input
    st.markdown(
        "<div style='margin-top: 2.2rem;'><a href='https://developer.nlr.gov/signup/' target='_blank'>Get a free API Key</a></div>",
        unsafe_allow_html=True)

email = st.text_input("Email Address", help="The email associated with your API key")

# --- Map & Coordinates Section ---
st.markdown("### 2. Location")
st.write("Search for a location using the magnifying glass icon on the map, or click anywhere to set the coordinates.")

# Initialize default coordinates (Denver, CO) in session state
if "lat" not in st.session_state:
    st.session_state.lat = 39.7410
if "lon" not in st.session_state:
    st.session_state.lon = -105.1702
if "zoom" not in st.session_state:
    st.session_state.zoom = 4

# Build the interactive Folium map
m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=st.session_state.zoom)
# Add a search bar to the map
Geocoder().add_to(m)
# Add a marker for the current selection
folium.Marker([st.session_state.lat, st.session_state.lon], tooltip="Current Selection").add_to(m)

# Render the map in Streamlit (this catches map clicks automatically).
# A stable key avoids remounting the map component on every rerun, and
# limiting returned_objects means a Streamlit rerun only fires when the
# click or zoom actually change, not on every pan/hover.
map_data = st_folium(
    m,
    height=400,
    width=700,
    key="location_map",
    returned_objects=["last_clicked", "zoom"],
)

# Update coordinates if the user clicked the map
if map_data and map_data.get("last_clicked"):
    st.session_state.lat = map_data["last_clicked"]["lat"]
    st.session_state.lon = map_data["last_clicked"]["lng"]

# Preserve the current zoom level so the map doesn't reset on rerun
if map_data and map_data.get("zoom"):
    st.session_state.zoom = map_data["zoom"]

col3, col4 = st.columns(2)
with col3:
    lat = st.number_input("Latitude", value=st.session_state.lat, format="%.4f")
with col4:
    lon = st.number_input("Longitude", value=st.session_state.lon, format="%.4f")

# --- Settings Section ---
st.markdown("### 3. Dataset Settings")
col5, col6 = st.columns(2)
with col5:
    resolution = st.selectbox("Time Resolution", ["15", "60"], format_func=lambda x: f"{x} Minutes")
with col6:
    year = st.selectbox("Year", ["2022", "2021", "2020", "2019", "2018"])

# --- Download Logic ---
if st.button("Fetch Data from NSRDB", type="primary"):
    if not api_key or not email:
        st.error("Please provide both an API Key and an Email Address.")
    else:
        # 1. Automatically route to the correct satellite dataset based on longitude
        if -180 <= lon <= -30:
            # Americas (GOES satellites)
            # Supports 5, 15, 30, 60 mins
            dataset_endpoint = "nsrdb-GOES-conus-v4-0-0-download.csv"

        elif -30 < lon <= 60:
            # Europe, Africa, Middle East (Meteosat MSG)
            # Supports 15, 30, 60 mins
            dataset_endpoint = "nsrdb-msg-v1-0-0-download.csv"

        else:
            # Asia / Pacific (Himawari)
            # Himawari supports 10, 30, 60 mins (but not 15)
            dataset_endpoint = "himawari7-download.csv"

        if dataset_endpoint == "himawari7-download.csv" and resolution == "15":
            st.error(
                "15-minute data is not available for this location (the Himawari dataset only supports 10, 30, or 60-minute resolution). "
                "Please choose a different time resolution or pick a location outside the Asia/Pacific region.")
        else:
            with st.spinner("Requesting data from NSRDB..."):

                # Construct the dynamic URL
                url = f"https://developer.nlr.gov/api/nsrdb/v2/solar/{dataset_endpoint}"

                params = {
                    "api_key": api_key,
                    "email": email,
                    "wkt": f"POINT({lon} {lat})",  # Longitude comes first
                    "names": year,
                    "interval": resolution,
                    "attributes": "ghi,dni,dhi,air_temperature,wind_speed,wind_direction,"
                                   "relative_humidity,surface_pressure,total_precipitable_water,"
                                   "solar_zenith_angle",
                    "utc": "true",  # Always request data in UTC
                    "leap_day": "false"
                }

                try:
                    response = requests.get(url, params=params)

                    if response.status_code == 200:
                        raw_df = parse_nsrdb_response(response.text)
                        csv_text, preview_df = build_output_csv(raw_df)
                        st.success(
                            f"Data successfully downloaded using the {dataset_endpoint.split('-')[1].upper()} dataset!")
                        st.dataframe(preview_df.head(10))

                        ghi_sum, calculated_sum, diff_pct = check_ghi_closure(raw_df)
                        st.write(
                            f"Cumulative GHI: {ghi_sum:,.0f} — Cumulative DHI + DNI·cos(zenith): "
                            f"{calculated_sum:,.0f} (difference: {diff_pct:.2f}%)")
                        if diff_pct > 1:
                            st.warning(
                                f"GHI closure check failed: cumulative GHI differs from DHI + DNI·cos(zenith) "
                                f"by {diff_pct:.2f}%, more than the 1% tolerance. This may indicate a data "
                                f"quality issue with the downloaded dataset.")

                        st.download_button(
                            label="Download full dataset as CSV",
                            data=csv_text.encode('utf-8'),
                            file_name=f"nsrdb_data_{lat}_{lon}_{year}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")

                except Exception as e:
                    st.error(f"An error occurred while making the request: {e}")