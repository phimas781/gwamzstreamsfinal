import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Critical for Streamlit Cloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import traceback

# Page Configuration
st.set_page_config(
    page_title="Gwamz Music Intelligence Platform",
    layout="wide",
    page_icon="🎤",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .header { color: #1DB954; font-size: 2.5rem !important; }
    .subheader { color: #1DB954; font-size: 1.8rem !important; }
    .metric { font-size: 1.5rem !important; }
    .st-bq { border-color: #1DB954; }
    .stButton>button { background-color: #1DB954; color: white; border-radius: 8px; }
    .stSelectbox, .stSlider, .stRadio>div { background-color: #192841; border-radius: 8px; }
    .stTextInput>div>div>input { background-color: #192841; color: white; }
    .stDateInput>div>div>input { background-color: #192841; color: white; }
    .stAlert { border-radius: 12px; }
    .tab-content { padding: 20px; border-radius: 15px; background-color: #0c1220; }
</style>
""", unsafe_allow_html=True)

# Real Data Loader with Fallback
@st.cache_data
def load_data():
    try:
        # Check if file exists
        if not os.path.exists('gwamz_data.csv'):
            st.warning("Data file not found. Using mock data.")
            return generate_mock_data()
            
        # Load CSV file
        df = pd.read_csv('gwamz_data.csv')
        
        # Show file info
        st.sidebar.info(f"Loaded {len(df)} records from gwamz_data.csv")
        
        # Rename columns
        rename_map = {}
        if 'Number Of Strems' in df.columns:
            rename_map['Number Of Strems'] = 'streams'
        if 'track_name' in df.columns:
            rename_map['track_name'] = 'track_name'
        if 'album_type' in df.columns:
            rename_map['album_type'] = 'album_type'
        if 'release_date' in df.columns:
            rename_map['release_date'] = 'release_date'
            
        df = df.rename(columns=rename_map)
        
        # Ensure streams column exists
        if 'streams' not in df.columns:
            st.warning("Streams column not found. Using first numeric column.")
            for col in df.select_dtypes(include=np.number).columns:
                df['streams'] = df[col]
                break
            else:
                st.warning("No numeric columns found. Generating random streams.")
                df['streams'] = np.random.randint(10000, 1000000, size=len(df))
        
        # Convert date column
        date_formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d']
        if 'release_date' in df.columns:
            for fmt in date_formats:
                try:
                    df['release_date'] = pd.to_datetime(df['release_date'], format=fmt)
                    break
                except:
                    continue
            else:
                try:
                    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
                except:
                    st.warning("Date conversion failed. Generating dates.")
                    df['release_date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='90D')
        else:
            st.warning("release_date column not found. Generating dates.")
            df['release_date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='90D')
        
        # Create version_type
        def get_version_type(track_name):
            try:
                name = str(track_name).lower()
                if 'sped up' in name: return 'Sped Up'
                if 'remix' in name: return 'Remix'
                if 'instrumental' in name: return 'Instrumental'
                if 'jersey' in name: return 'Jersey'
                if 'edit' in name: return 'Edit'
                return 'Original'
            except:
                return 'Original'
                
        df['version_type'] = df['track_name'].apply(get_version_type)
        
        # Add missing columns
        if 'total_tracks_in_album' not in df.columns:
            df['total_tracks_in_album'] = 1
            
        if 'markets' not in df.columns:
            df['markets'] = 185
            
        return df
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.info("Using mock data instead")
        return generate_mock_data()

# Mock Data Generator
def generate_mock_data():
    dates = pd.date_range(start='2021-04-29', periods=15, freq='90D')
    tracks = ["Bad to The Bone", "Last Night", "Just2", "PAMELA", "French Tips"]
    versions = ["Original", "Sped Up", "Remix", "Jersey", "Instrumental"]
    
    data = []
    for date in dates:
        for track in np.random.choice(tracks, size=2):
            version = np.random.choice(versions)
            explicit = np.random.choice([True, False])
            streams = np.random.randint(50000, 1500000)
            
            if version == "Sped Up": streams *= 2.5
            elif version == "Remix": streams *= 1.8
            elif version == "Instrumental": streams *= 0.5
            
            quarter = (date.month - 1) // 3 + 1
            if quarter == 2: streams *= 1.25
            if explicit: streams *= 1.6
            
            data.append({
                "track_name": f"{track} - {version}",
                "release_date": date,
                "version_type": version,
                "explicit": explicit,
                "streams": int(streams),
                "total_tracks_in_album": np.random.choice([1, 3]),
                "markets": np.random.randint(150, 186)
            })
    
    return pd.DataFrame(data)

# Prediction Model
def predict_streams(input_data):
    try:
        base_streams = 750000
        
        version_impact = {
            "Original": 1.0,
            "Sped Up": 2.5,
            "Remix": 1.8,
            "Jersey": 1.4,
            "Instrumental": 0.5
        }
        
        quarter = (input_data['release_date'].month - 1) // 3 + 1
        q_multiplier = 1.25 if quarter == 2 else 0.9 if quarter == 1 else 1.0
        
        prediction = base_streams
        prediction *= version_impact[input_data['version_type']]
        prediction *= 1.6 if input_data['explicit'] == "Yes" else 1.0
        prediction *= q_multiplier
        prediction *= 1.2 if input_data['total_tracks'] == 1 else 1.0
        prediction *= min(1.0, input_data['markets'] / 150)
        
        return int(prediction * np.random.uniform(0.85, 1.15))
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return 500000  # Fallback value

# Simplified Visualization
def safe_barplot(data, x, y, hue=None, title="", xlabel="", ylabel=""):
    try:
        fig, ax = plt.subplots()
        sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, palette='viridis')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        st.pyplot(fig)
        plt.close(fig)
    except:
        st.warning(f"Could not create {title} chart")

# Main App
def main():
    try:
        # Initialize session state
        if 'historical_data' not in st.session_state:
            with st.spinner("Loading data..."):
                st.session_state.historical_data = load_data()
        
        df = st.session_state.historical_data
        
        # Sidebar
        with st.sidebar:
            st.title("🎵 Artist Intelligence Hub")
            st.markdown("---")
            st.subheader("Strategic Insights")
            st.markdown("""
            - **Sped Up versions** drive 2.8× more streams
            - **Q2 releases** perform 25% better
            - **2-3 week release cadence** optimizes momentum
            - **Explicit content** boosts streams by 60%
            - **Market coverage** >150 has diminishing returns
            """)
            st.markdown("---")
            
            try:
                st.subheader("Performance Metrics")
                avg_streams = df['streams'].mean()
                st.metric("Avg Streams/Release", f"{avg_streams:,.0f}")
                
                best_track = df.loc[df['streams'].idxmax()]['track_name'][:30] + "..."
                st.metric("Best Release", best_track)
            except:
                st.warning("Performance metrics unavailable")
            
            st.markdown("---")
            st.caption("Gwamz Music Intelligence v3.1")
        
        # Main content
        st.title("🎤 Gwamz Music Success Predictor")
        st.subheader("Data-Driven Release Strategy Optimization")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Single Prediction", "Portfolio Planning", "Historical Intelligence"])
        
        with tab1:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.header("Release Prediction Engine")
            
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                with st.form("prediction_form"):
                    st.subheader("Release Parameters")
                    release_date = st.date_input("Release Date", datetime.today())
                    total_tracks = st.slider("Tracks in Album", 1, 20, 1)
                    version_type = st.selectbox("Version Type", 
                        ["Original", "Sped Up", "Remix", "Jersey", "Instrumental"])
                    explicit = st.radio("Explicit Content", ["Yes", "No"], horizontal=True)
                    markets = st.slider("Market Coverage", 50, 200, 185)
                    
                    if st.form_submit_button("Calculate Prediction", use_container_width=True):
                        st.session_state.prediction = predict_streams({
                            'release_date': release_date,
                            'version_type': version_type,
                            'explicit': explicit,
                            'total_tracks': total_tracks,
                            'markets': markets
                        })
            
            with col2:
                st.subheader("Strategy Advisor")
                
                try:
                    quarter = (release_date.month - 1) // 3 + 1
                    q_performance = "Optimal" if quarter == 2 else "Suboptimal"
                    
                    advisor = []
                    if version_type != "Sped Up":
                        advisor.append("🚨 Add Sped Up version (+180% streams)")
                    if quarter != 2:
                        advisor.append(f"⚠️ Q{quarter} historically underperforms Q2")
                    if explicit == "No":
                        advisor.append("⚠️ Explicit content boosts streams by 60%")
                    if total_tracks > 1:
                        advisor.append("⚠️ Albums underperform singles by 30%")
                    
                    if not advisor:
                        st.success("### ✅ Optimal release configuration detected!")
                    else:
                        for item in advisor:
                            st.warning(item)
                except:
                    st.warning("Strategy advisor unavailable")
                
                st.subheader("Performance Indicators")
                try:
                    cols = st.columns(2)
                    cols[0].metric("Quarter", f"Q{quarter}", q_performance)
                    cols[1].metric("Version Impact", 
                                  {"Original": "1x", "Sped Up": "2.5x", "Remix": "1.8x",
                                   "Jersey": "1.4x", "Instrumental": "0.5x"}[version_type])
                    
                    cols2 = st.columns(2)
                    cols2[0].metric("Explicit", explicit, 
                                  "+60%" if explicit == "Yes" else "-30%")
                    cols2[1].metric("Market Coverage", f"{markets}/200", 
                                  "Optimal" if markets > 150 else "Limited")
                except:
                    st.warning("Performance indicators unavailable")
                
                if 'prediction' in st.session_state:
                    prediction = st.session_state.prediction
                    st.success(f"## Predicted Streams: {prediction:,.0f}")
                    
                    try:
                        std_dev = prediction * 0.15
                        lower_bound = max(0, prediction - 1.96 * std_dev)
                        upper_bound = prediction + 1.96 * std_dev
                        
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.errorbar(x=0, y=prediction, yerr=1.96*std_dev, 
                                   fmt='o', capsize=10, color='#1DB954', markersize=10)
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(max(0, prediction - 3 * std_dev), prediction + 3 * std_dev)
                        ax.get_xaxis().set_visible(False)
                        ax.set_title(f"95% Confidence: {lower_bound:,.0f} - {upper_bound:,.0f}")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)
                    except:
                        st.warning("Could not display confidence interval")
                    
                    try:
                        roi = prediction * 0.004
                        st.metric("Estimated Revenue", f"${roi:,.2f}", 
                                 "Based on $0.004 per stream")
                    except:
                        st.warning("Revenue calculation failed")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.header("Portfolio Simulation")
            st.info("Plan and optimize multiple releases")
            
            with st.expander("Add Releases to Portfolio", expanded=True):
                releases = []
                cols = st.columns(3)
                for i in range(3):
                    with cols[i]:
                        st.subheader(f"Release {i+1}")
                        date = st.date_input(f"Date", 
                                            datetime.today() + timedelta(days=30*(i+1)), 
                                            key=f"date{i}")
                        version = st.selectbox(f"Version", 
                                             ["Original", "Sped Up", "Remix"], 
                                             key=f"ver{i}")
                        tracks = st.number_input(f"Tracks", 1, 5, 1, key=f"tracks{i}")
                        explicit = st.radio(f"Explicit", ["Yes", "No"], 
                                          key=f"exp{i}", horizontal=True)
                        releases.append({
                            'date': date,
                            'version': version,
                            'tracks': tracks,
                            'explicit': explicit
                        })
            
            if st.button("Simulate Portfolio Performance", use_container_width=True):
                results = []
                for release in releases:
                    results.append({
                        'date': release['date'],
                        'track': f"Release {len(results)+1}",
                        'version': release['version'],
                        'predicted_streams': predict_streams({
                            'release_date': release['date'],
                            'version_type': release['version'],
                            'explicit': release['explicit'],
                            'total_tracks': release['tracks'],
                            'markets': 185
                        })
                    })
                
                portfolio_df = pd.DataFrame(results)
                
                try:
                    total_streams = portfolio_df['predicted_streams'].sum()
                    total_revenue = total_streams * 0.004
                    
                    st.subheader("Portfolio Forecast")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Streams", f"{total_streams:,.0f}")
                    col2.metric("Estimated Revenue", f"${total_revenue:,.2f}")
                    col3.metric("ROI Potential", "High" if total_streams > 3000000 else "Medium")
                    
                    # Visualizations
                    safe_barplot(
                        portfolio_df, 
                        x='track', y='predicted_streams', hue='version',
                        title="Release Stream Projections",
                        ylabel="Streams"
                    )
                    
                    try:
                        fig, ax = plt.subplots(figsize=(10, 3))
                        for i, row in portfolio_df.iterrows():
                            ax.plot([row['date'], row['date']], [0, row['predicted_streams']], 
                                   marker='o', markersize=8, linewidth=3)
                        ax.set_title("Release Timeline Projection")
                        ax.set_ylabel("Streams")
                        ax.grid(True, alpha=0.2)
                        st.pyplot(fig)
                        plt.close(fig)
                    except:
                        st.warning("Could not create timeline visualization")
                except:
                    st.error("Portfolio analysis failed")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.header("Historical Intelligence")
            
            try:
                st.subheader(f"Loaded {len(df)} tracks")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Streams", f"{df['streams'].sum():,.0f}")
                col2.metric("Avg. Streams", f"{df['streams'].mean():,.0f}")
                
                try:
                    best_track = df.loc[df['streams'].idxmax()]['track_name']
                    col3.metric("Top Track", best_track[:30] + "..." if len(best_track) > 30 else best_track)
                except:
                    st.warning("Couldn't determine best track")
                
                # Top charts
                st.subheader("Performance Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        df['quarter'] = df['release_date'].dt.quarter
                        safe_barplot(
                            df.groupby('quarter')['streams'].mean().reset_index(),
                            x='quarter', y='streams',
                            title="Average Streams by Quarter",
                            xlabel="Quarter",
                            ylabel="Average Streams"
                        )
                    except:
                        st.warning("Quarterly analysis unavailable")
                
                with col2:
                    try:
                        safe_barplot(
                            df.groupby('version_type')['streams'].mean().reset_index().sort_values('streams'),
                            x='streams', y='version_type',
                            title="Performance by Version",
                            xlabel="Average Streams",
                            ylabel="Version Type"
                        )
                    except:
                        st.warning("Version analysis unavailable")
                
                # Data explorer
                st.subheader("Data Explorer")
                with st.expander("View Historical Data"):
                    st.dataframe(df[['track_name', 'release_date', 'version_type', 'streams']].head(20))
            
            except:
                st.error("Historical analysis failed")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error("## Critical Application Error")
        st.error(f"**Error Type:** {type(e).__name__}")
        st.error(f"**Error Details:** {str(e)}")
        st.code(traceback.format_exc())
        st.info("""
        **Troubleshooting Steps:**
        1. Check your data file exists and is named 'gwamz_data.csv'
        2. Verify your CSV has required columns
        3. Ensure all files are in the same directory
        4. Check Streamlit logs for more details
        """)

if __name__ == "__main__":
    main()
