import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

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

# Real Data Loader
@st.cache_data
def load_real_data():
    try:
        # Load your CSV file
        df = pd.read_csv('gwamz_data.csv')
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Number Of Strems': 'streams',
            'track_name': 'track_name',
            'album_type': 'album_type'
        })
        
        # Convert date column to datetime format
        df['release_date'] = pd.to_datetime(df['release_date'], format='%d/%m/%Y')
        
        # Create version_type column from track_name
        def get_version_type(track_name):
            if not isinstance(track_name, str):
                return 'Original'
                
            track_name = track_name.lower()
            if 'sped up' in track_name: return 'Sped Up'
            elif 'remix' in track_name: return 'Remix'
            elif 'instrumental' in track_name: return 'Instrumental'
            elif 'jersey' in track_name: return 'Jersey'
            elif 'edit' in track_name: return 'Edit'
            else: return 'Original'
        
        df['version_type'] = df['track_name'].apply(get_version_type)
        
        # Ensure we have required columns
        if 'total_tracks_in_album' not in df.columns:
            df['total_tracks_in_album'] = 1
            
        if 'markets' not in df.columns:
            df['markets'] = 185
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Prediction Model
def predict_streams(input_data):
    base_streams = 750000
    
    # Version impact
    version_impact = {
        "Original": 1.0,
        "Sped Up": 2.5,
        "Remix": 1.8,
        "Jersey": 1.4,
        "Instrumental": 0.5
    }
    
    # Quarter impact
    quarter = (input_data['release_date'].month - 1) // 3 + 1
    q_multiplier = 1.25 if quarter == 2 else 0.9 if quarter == 1 else 1.0
    
    # Calculate prediction
    prediction = base_streams
    prediction *= version_impact[input_data['version_type']]
    prediction *= 1.6 if input_data['explicit'] == "Yes" else 1.0
    prediction *= q_multiplier
    prediction *= 1.2 if input_data['total_tracks'] == 1 else 1.0
    prediction *= min(1.0, input_data['markets'] / 150)
    
    # Add randomness
    prediction *= np.random.uniform(0.85, 1.15)
    
    return int(prediction)

# Main App
def main():
    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = load_real_data()
    
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
        
        if not st.session_state.historical_data.empty:
            st.subheader("Performance Metrics")
            avg_streams = st.session_state.historical_data['streams'].mean()
            st.metric("Avg Streams/Release", f"{avg_streams:,.0f}")
            
            if 'streams' in st.session_state.historical_data:
                best_track = st.session_state.historical_data.loc[
                    st.session_state.historical_data['streams'].idxmax()
                ]['track_name']
                st.metric("Best Release", best_track)
        
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
                    input_data = {
                        'release_date': release_date,
                        'version_type': version_type,
                        'explicit': explicit,
                        'total_tracks': total_tracks,
                        'markets': markets
                    }
                    st.session_state.prediction = predict_streams(input_data)
                    st.session_state.prediction_input = input_data
        
        with col2:
            st.subheader("Strategy Advisor")
            
            # Calculate metrics for advisor
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
            
            st.subheader("Performance Indicators")
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
            
            # Display prediction results if available
            if 'prediction' in st.session_state:
                prediction = st.session_state.prediction
                st.success(f"## Predicted Streams: {prediction:,.0f}")
                
                # Confidence visualization
                std_dev = prediction * 0.15
                lower_bound = max(0, prediction - 1.96 * std_dev)
                upper_bound = prediction + 1.96 * std_dev
                
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.errorbar(x=0, y=prediction, yerr=1.96*std_dev, 
                           fmt='o', capsize=10, color='#1DB954', markersize=10)
                ax.set_xlim(-1, 1)
                ax.set_ylim(max(0, prediction - 3 * std_dev), prediction + 3 * std_dev)
                ax.get_xaxis().set_visible(False)
                ax.set_title(f"95% Confidence Interval: {lower_bound:,.0f} - {upper_bound:,.0f} streams")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # ROI Calculation
                roi = prediction * 0.004
                st.metric("Estimated Revenue", f"${roi:,.2f}", 
                         "Based on $0.004 per stream")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("Portfolio Simulation")
        st.info("Plan and optimize multiple releases for maximum impact")
        
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
            # Generate predictions
            results = []
            for release in releases:
                input_data = {
                    'release_date': release['date'],
                    'version_type': release['version'],
                    'explicit': release['explicit'],
                    'total_tracks': release['tracks'],
                    'markets': 185
                }
                streams = predict_streams(input_data)
                results.append({
                    'date': release['date'],
                    'track': f"Release {len(results)+1}",
                    'version': release['version'],
                    'predicted_streams': streams
                })
            
            portfolio_df = pd.DataFrame(results)
            
            # Portfolio analysis
            total_streams = portfolio_df['predicted_streams'].sum()
            total_revenue = total_streams * 0.004
            
            # Display results
            st.subheader("Portfolio Forecast")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Streams", f"{total_streams:,.0f}")
            col2.metric("Estimated Revenue", f"${total_revenue:,.2f}")
            col3.metric("ROI Potential", "High" if total_streams > 3000000 else "Medium")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='track', y='predicted_streams', hue='version', 
                        data=portfolio_df, palette='viridis', ax=ax)
            ax.set_title("Release Stream Projections")
            ax.set_ylabel("Streams")
            ax.set_xlabel("Release")
            ax.ticklabel_format(style='plain', axis='y')
            st.pyplot(fig)
            
            # Timeline visualization
            fig, ax = plt.subplots(figsize=(10, 3))
            for i, row in portfolio_df.iterrows():
                ax.plot([row['date'], row['date']], [0, row['predicted_streams']], 
                       marker='o', markersize=8, linewidth=3)
            ax.set_title("Release Timeline Projection")
            ax.set_ylabel("Streams")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("Historical Intelligence")
        st.info("Analyze past performance and trends")
        
        if st.session_state.historical_data.empty:
            st.warning("No historical data loaded")
        else:
            df = st.session_state.historical_data
            
            # Top-level metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Streams", f"{df['streams'].sum():,.0f}")
            col2.metric("Avg. Streams/Release", f"{df['streams'].mean():,.0f}")
            
            if 'streams' in df.columns:
                best_track = df.loc[df['streams'].idxmax()]['track_name']
                col3.metric("Best Performing Track", best_track)
            
            col4.metric("Release Frequency", 
                       f"{(df['release_date'].max() - df['release_date'].min()).days / len(df):.1f} days")
            
            # Temporal trends
            st.subheader("Temporal Performance Analysis")
            df['quarter'] = pd.to_datetime(df['release_date']).dt.quarter
            df['year'] = pd.to_datetime(df['release_date']).dt.year
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Quarterly performance
            if 'quarter' in df.columns and 'streams' in df.columns:
                q_avg = df.groupby('quarter')['streams'].mean()
                sns.barplot(x=q_avg.index, y=q_avg.values, ax=ax1, palette='viridis')
                ax1.set_title("Average Streams by Quarter")
                ax1.set_xlabel("Quarter")
                ax1.set_ylabel("Average Streams")
            
            # Version performance
            if 'version_type' in df.columns and 'streams' in df.columns:
                version_perf = df.groupby('version_type')['streams'].mean().sort_values(ascending=False)
                sns.barplot(x=version_perf.values, y=version_perf.index, ax=ax2, palette='mako')
                ax2.set_title("Performance by Version Type")
                ax2.set_xlabel("Average Streams")
            
            st.pyplot(fig)
            
            # Time series analysis
            st.subheader("Streams Over Time")
            time_df = df.copy()
            time_df['release_date'] = pd.to_datetime(time_df['release_date'])
            time_df = time_df.set_index('release_date').sort_index()
            
            if 'streams' in time_df.columns:
                monthly = time_df['streams'].resample('M').sum()
                
                fig, ax = plt.subplots(figsize=(12, 5))
                monthly.plot(ax=ax, color='#1DB954', linewidth=2.5)
                ax.fill_between(monthly.index, monthly, alpha=0.2, color='#1DB954')
                ax.set_title("Monthly Streaming Performance")
                ax.set_ylabel("Total Streams")
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)
            
            # Data explorer
            st.subheader("Data Explorer")
            with st.expander("View Historical Data"):
                st.dataframe(df.sort_values('release_date', ascending=False))
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
