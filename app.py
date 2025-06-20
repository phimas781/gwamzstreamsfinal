import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'show_historical' not in st.session_state:
    st.session_state.show_historical = False

# Page Configuration
st.set_page_config(
    page_title="Gwamz Music Predictor",
    layout="wide",
    page_icon="🎤",
)

# Custom Styling
st.markdown("""
<style>
    .stApp { background: #0e1117; color: white; }
    h1, h2, h3 { color: #1DB954 !important; }
    .stButton>button { background: #1DB954; color: white; border-radius: 8px; }
    .stAlert { border-left: 4px solid #1DB954; }
    .metric { font-size: 1.4rem; }
    .card { background: #192841; border-radius: 10px; padding: 20px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Load Data - Robust Implementation
def load_data():
    """Load data with multiple fallback options"""
    try:
        # Try possible file names
        for filename in ['gwamz_data.csv', 'gwamz_spotify_data.csv', 'gwamz_spotify_data (2).csv']:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                
                # Standardize column names
                col_map = {}
                if 'Number Of Strems' in df.columns:
                    col_map['Number Of Strems'] = 'streams'
                if 'track_name' in df.columns:
                    col_map['track_name'] = 'track_name'
                if 'release_date' in df.columns:
                    col_map['release_date'] = 'release_date'
                    
                df = df.rename(columns=col_map)
                return df
                
        # If no files found, create sample data
        return pd.DataFrame({
            'track_name': ['Last Night', 'Just2', 'PAMELA'],
            'release_date': ['2023-03-16', '2024-04-18', '2023-10-12'],
            'streams': [2951075, 1157082, 766818]
        })
        
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame({
            'track_name': ['Sample 1', 'Sample 2'],
            'release_date': ['2023-01-01', '2023-02-01'],
            'streams': [500000, 750000]
        })

# Prediction Model (Rule-based)
def predict_streams(release_date, version_type, explicit, total_tracks, markets):
    """Simple prediction model based on business rules"""
    base = 750000
    
    # Version multipliers
    version_impact = {
        "Original": 1.0,
        "Sped Up": 2.5,
        "Remix": 1.8,
        "Jersey": 1.4,
        "Instrumental": 0.5
    }
    
    # Quarter impact (Q2 is best)
    quarter = (release_date.month - 1) // 3 + 1
    q_multiplier = 1.25 if quarter == 2 else 1.0
    
    # Calculate prediction
    prediction = base
    prediction *= version_impact.get(version_type, 1.0)
    prediction *= 1.6 if explicit == "Yes" else 1.0
    prediction *= q_multiplier
    prediction *= 1.2 if total_tracks == 1 else 1.0
    prediction *= min(1.0, markets / 150)  # Market saturation
    
    # Add some randomness
    prediction *= np.random.uniform(0.9, 1.1)
    
    return int(prediction)

# Main App
def main():
    st.title("🎤 Gwamz Music Success Predictor")
    st.write("Forecast your song performance using data-driven insights")
    
    # Load data
    df = load_data()
    
    # Convert dates
    try:
        df['release_date'] = pd.to_datetime(df['release_date'])
    except:
        st.warning("Could not parse dates in historical data")
    
    # Sidebar
    with st.sidebar:
        st.header("Artist Intelligence Hub")
        st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png", width=150)
        
        st.subheader("Key Insights")
        st.markdown("""
        - 🚀 **Sped Up** versions get 2.5× more streams
        - 🌞 **Q2 releases** (Apr-Jun) perform best
        - ⚠️ **Explicit** content boosts streams by 60%
        - 💽 **Singles** outperform albums
        """)
        
        if not df.empty:
            st.subheader("Historical Stats")
            st.metric("Total Songs", len(df))
            st.metric("Avg Streams", f"{df['streams'].mean():,.0f}")
        
        st.markdown("---")
        st.caption("Gwamz Analytics v1.0")
    
    # Main Content - Two Tabs
    tab1, tab2 = st.tabs(["Song Predictor", "Historical Data"])
    
    with tab1:
        st.header("Predict New Song Performance")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Release Details")
                release_date = st.date_input("Release Date", datetime.today())
                total_tracks = st.slider("Tracks in Album", 1, 20, 1,
                                         help="Singles (1 track) perform best")
                markets = st.slider("Market Coverage", 50, 200, 185)
                
            with col2:
                st.subheader("Song Features")
                version_type = st.selectbox("Version Type", 
                    ["Original", "Sped Up", "Remix", "Jersey", "Instrumental"])
                explicit = st.radio("Explicit Content", ["Yes",
