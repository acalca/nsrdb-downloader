import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw
import requests
import pandas as pd
import numpy as np
import math
import io

st.set_page_config(page_title="NSRDB Downloader", layout="centered")

# --- Static map (no third-party JS map widget - see note in the Location
# section below for why). Voyager is CARTO's colorful, Google-Maps-like
# style (vs. the plainer "light_all"); @2x is the retina/high-DPI tile
# variant for a crisper render. Still free, no API key needed, same
# attribution terms as before.
TILE_SIZE = 512
TILE_URL = "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png"
MAP_WIDTH, MAP_HEIGHT = 700, 400
MIN_ZOOM, MAX_ZOOM = 2, 17


def lonlat_to_world_px(lon, lat, zoom):
    """Web Mercator projection: geographic coords -> pixel coords in the full world map at this zoom."""
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(max(min(lat, 85.05), -85.05))
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y


def world_px_to_lonlat(x, y, zoom):
    """Inverse of lonlat_to_world_px: pixel coords in the world map -> geographic coords."""
    n = 2 ** zoom
    lon = x / (n * TILE_SIZE) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / (n * TILE_SIZE))))
    return math.degrees(lat_rad), lon


@st.cache_data(show_spinner=False, max_entries=1000)
def fetch_tile(zoom, x, y):
    """Fetch and cache a single map tile image."""
    n = 2 ** zoom
    x = x % n  # wrap around the antimeridian
    response = requests.get(
        TILE_URL.format(z=zoom, x=x, y=y),
        headers={"User-Agent": "nsrdb-downloader-streamlit-app"},
        timeout=10,
    )
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def render_map_image(center_lat, center_lon, zoom, width, height):
    """Stitch tiles into a single image centered on (center_lat, center_lon), with a marker at the center."""
    center_x, center_y = lonlat_to_world_px(center_lon, center_lat, zoom)
    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    n = 2 ** zoom

    canvas = Image.new("RGB", (width, height), color=(221, 221, 221))
    first_tx, first_ty = int(top_left_x // TILE_SIZE), int(top_left_y // TILE_SIZE)
    last_tx, last_ty = int((top_left_x + width) // TILE_SIZE), int((top_left_y + height) // TILE_SIZE)

    for tx in range(first_tx, last_tx + 1):
        for ty in range(first_ty, last_ty + 1):
            if ty < 0 or ty >= n:
                continue
            try:
                tile_img = fetch_tile(zoom, tx, ty)
            except Exception:
                continue
            canvas.paste(tile_img, (int(tx * TILE_SIZE - top_left_x), int(ty * TILE_SIZE - top_left_y)))

    draw = ImageDraw.Draw(canvas)
    cx, cy = width // 2, height // 2
    draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(220, 30, 30), outline=(0, 0, 0), width=2)

    return canvas


DOUBLE_CLICK_MS = 500
DOUBLE_CLICK_PIXEL_TOLERANCE = 6


def _on_map_click():
    """Convert the clicked pixel back to lat/lon, using the view that was active when it was rendered.

    Also detects a double-click (two clicks close in time and position) and
    treats it as "zoom in here", the conventional map gesture - there's no
    mouse-wheel event available through this component.
    """
    click = st.session_state.location_map_img
    if not click:
        return

    last_click = st.session_state.get("_last_map_click")
    is_double_click = (
        last_click is not None
        and click["unix_time"] - last_click["unix_time"] <= DOUBLE_CLICK_MS
        and abs(click["x"] - last_click["x"]) <= DOUBLE_CLICK_PIXEL_TOLERANCE
        and abs(click["y"] - last_click["y"]) <= DOUBLE_CLICK_PIXEL_TOLERANCE
    )
    st.session_state._last_map_click = {"unix_time": click["unix_time"], "x": click["x"], "y": click["y"]}

    if is_double_click:
        # Zoom in on the point the first click of this pair already
        # selected, rather than recomputing lat/lon from this click's
        # pixel - the view may have already shifted after that first click.
        st.session_state.zoom = min(MAX_ZOOM, st.session_state.zoom + 2)
        return

    center_x, center_y = lonlat_to_world_px(st.session_state.lon, st.session_state.lat, st.session_state.zoom)
    top_left_x = center_x - MAP_WIDTH / 2
    top_left_y = center_y - MAP_HEIGHT / 2
    lat, lon = world_px_to_lonlat(top_left_x + click["x"], top_left_y + click["y"], st.session_state.zoom)
    st.session_state.lat = lat
    st.session_state.lon = lon


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
st.write("Search for a place, click the map, or type coordinates directly below.")

if "lat" not in st.session_state:
    st.session_state.lat = 38.199636821203164
if "lon" not in st.session_state:
    st.session_state.lon = -7.498110901770388
if "zoom" not in st.session_state:
    st.session_state.zoom = 10

# The map is a plain server-rendered image (stitched from map tiles), not an
# embedded interactive JS map widget (e.g. streamlit_folium/Leaflet). That
# earlier approach kept misbehaving - it only truly initializes once and
# largely ignores further updates, making pan/zoom/click state fight against
# Streamlit's rerun model. A static image + streamlit_image_coordinates'
# native on_click callback (Streamlit's regular, well-tested widget-callback
# protocol) avoids that entirely: every rerun renders a fresh image for the
# current center/zoom, no client-side map state to keep in sync at all.
def _search_place():
    query = st.session_state.get("place_query")
    if not query:
        return
    try:
        geocode_response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "nsrdb-downloader-streamlit-app"},
            timeout=10,
        )
        geocode_response.raise_for_status()
        results = geocode_response.json()
        if results:
            st.session_state.lat = float(results[0]["lat"])
            st.session_state.lon = float(results[0]["lon"])
            st.session_state["_place_search_error"] = None
        else:
            st.session_state["_place_search_error"] = f"No results found for '{query}'."
    except Exception as e:
        st.session_state["_place_search_error"] = f"Place search failed: {e}"


search_col, button_col = st.columns([4, 1])
with search_col:
    # on_change fires on Enter (or losing focus), so searching no longer
    # requires pressing Enter *and then* clicking Search - either on its own
    # triggers the same search.
    st.text_input(
        "Search for a place", key="place_query", placeholder="e.g. Vilnius, Lithuania",
        label_visibility="collapsed", on_change=_search_place)
with button_col:
    st.button("Search", use_container_width=True, on_click=_search_place)

if st.session_state.get("_place_search_error"):
    st.warning(st.session_state["_place_search_error"])
    st.session_state["_place_search_error"] = None

zoom_out_col, zoom_in_col, _ = st.columns([1, 1, 4])
with zoom_out_col:
    if st.button("➖ Zoom out", use_container_width=True):
        st.session_state.zoom = max(MIN_ZOOM, st.session_state.zoom - 1)
with zoom_in_col:
    if st.button("➕ Zoom in", use_container_width=True):
        st.session_state.zoom = min(MAX_ZOOM, st.session_state.zoom + 1)

map_image = render_map_image(
    st.session_state.lat, st.session_state.lon, st.session_state.zoom, MAP_WIDTH, MAP_HEIGHT)
streamlit_image_coordinates(
    map_image, key="location_map_img", width=MAP_WIDTH, height=MAP_HEIGHT, on_click=_on_map_click)
st.caption(
    "Click the map to set the location, double-click to zoom in there (no mouse-wheel zoom - "
    "use the buttons or double-click instead). Map tiles © CARTO, data © OpenStreetMap contributors.")

col3, col4 = st.columns(2)
with col3:
    lat = st.number_input("Latitude", key="lat", format="%.4f")
with col4:
    lon = st.number_input("Longitude", key="lon", format="%.4f")

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