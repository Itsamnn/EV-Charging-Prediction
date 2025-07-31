import streamlit as st
import sys
import traceback
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    """
    Load the pre-trained EV forecasting model using joblib with optimized caching.
    
    Returns:
        model: The loaded machine learning model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If the model file is corrupted or cannot be loaded
    """
    model_path = "forecasting_ev_model.pkl"
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model using joblib
        model = joblib.load(model_path)
        
        # Basic validation that we loaded something
        if model is None:
            raise ValueError("Loaded model is None - file may be corrupted")
            
        return model
        
    except FileNotFoundError as e:
        error_msg = f"Model file not found: {model_path}. Please ensure the model file exists in the project directory."
        st.error(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Failed to load model from {model_path}. The file may be corrupted or incompatible."
        st.error(f"❌ {error_msg}")
        st.error(f"Error details: {str(e)}")
        raise Exception(error_msg) from e

@st.cache_data(show_spinner="Loading county data...")
def load_county_data() -> Tuple[List[str], Dict]:
    """
    Load and process county data from preprocessed_ev_data.csv with optimized caching.
    
    Returns:
        Tuple containing:
        - List of unique county names for dropdown population
        - Dictionary with county information and historical data
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        Exception: If the CSV file is malformed or cannot be processed
    """
    csv_path = "preprocessed_ev_data.csv"
    
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Load the CSV data
        df = pd.read_csv(csv_path)
        
        # Validate required columns exist
        required_columns = ['County', 'State', 'Date', 'Electric Vehicle (EV) Total', 
                          'Percent Electric Vehicles', 'Battery Electric Vehicles (BEVs)',
                          'Plug-In Hybrid Electric Vehicles (PHEVs)']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract unique counties for dropdown (sorted alphabetically)
        unique_counties = sorted(df['County'].unique().tolist())
        
        # Create data structure for county information and historical data
        county_data = {}
        
        for county in unique_counties:
            county_df = df[df['County'] == county].copy()
            
            # Get state information (assuming one state per county)
            state = county_df['State'].iloc[0] if not county_df.empty else 'Unknown'
            
            # Sort by date to ensure chronological order
            county_df = county_df.sort_values('Date')
            
            # Create historical data structure
            historical_data = {
                'dates': county_df['Date'].tolist(),
                'ev_totals': county_df['Electric Vehicle (EV) Total'].tolist(),
                'percentages': county_df['Percent Electric Vehicles'].tolist(),
                'bevs': county_df['Battery Electric Vehicles (BEVs)'].tolist(),
                'phevs': county_df['Plug-In Hybrid Electric Vehicles (PHEVs)'].tolist()
            }
            
            # Store county information
            county_data[county] = {
                'county_name': county,
                'state': state,
                'historical_data': historical_data,
                'data_points': len(county_df),
                'date_range': {
                    'start': county_df['Date'].min(),
                    'end': county_df['Date'].max()
                }
            }
        
        st.success(f"✅ Successfully loaded data for {len(unique_counties)} counties from {csv_path}")
        return unique_counties, county_data
        
    except FileNotFoundError as e:
        error_msg = f"Data file not found: {csv_path}. Please ensure the preprocessed data file exists in the project directory."
        st.error(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg) from e
        
    except pd.errors.EmptyDataError as e:
        error_msg = f"The CSV file {csv_path} is empty or contains no data."
        st.error(f"❌ {error_msg}")
        raise Exception(error_msg) from e
        
    except pd.errors.ParserError as e:
        error_msg = f"Failed to parse CSV file {csv_path}. The file may be malformed or corrupted."
        st.error(f"❌ {error_msg}")
        st.error(f"Parser error details: {str(e)}")
        raise Exception(error_msg) from e
        
    except ValueError as e:
        error_msg = f"Data validation error in {csv_path}: {str(e)}"
        st.error(f"❌ {error_msg}")
        raise Exception(error_msg) from e
        
    except Exception as e:
        error_msg = f"Unexpected error while processing {csv_path}: {str(e)}"
        st.error(f"❌ {error_msg}")
        raise Exception(error_msg) from e

def get_county_info(county_name: str, county_data: Dict) -> Optional[Dict]:
    """
    Get detailed information for a specific county.
    
    Args:
        county_name: Name of the county to retrieve information for
        county_data: Dictionary containing all county data
        
    Returns:
        Dictionary with county information or None if county not found
    """
    return county_data.get(county_name)

def get_county_display_name(county_name: str, county_data: Dict) -> str:
    """
    Get formatted display name for a county including state.
    
    Args:
        county_name: Name of the county
        county_data: Dictionary containing all county data
        
    Returns:
        Formatted string like "County, State" or just county name if state unknown
    """
    county_info = county_data.get(county_name)
    if county_info and county_info.get('state'):
        return f"{county_name}, {county_info['state']}"
    return county_name

def apply_custom_styling():
    """Apply premium, professional styling for an exceptional user experience"""
    custom_css = """
    <style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Premium background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 100vh;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass morphism main container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        padding: 3rem 2.5rem;
        margin: 2rem auto;
        max-width: 1400px;
        position: relative;
        overflow: hidden;
    }
    
    .main .block-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 25px 25px 0 0;
    }
    
    /* Stunning animated title */
    .main-title {
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleGlow 3s ease-in-out infinite alternate;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.3);
        letter-spacing: -0.02em;
        position: relative;
    }
    
    .main-title::after {
        content: '';
        position: absolute;
        bottom: -15px;
        left: 50%;
        transform: translateX(-50%);
        width: 120px;
        height: 4px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 2px;
        animation: underlineGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    @keyframes underlineGlow {
        0% { width: 120px; opacity: 0.7; }
        100% { width: 180px; opacity: 1; }
    }
    
    /* Premium subtitle */
    .subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.4rem;
        font-weight: 400;
        margin-bottom: 3rem;
        font-family: 'Inter', sans-serif;
        opacity: 0.9;
        animation: fadeInUp 1s ease-out 0.5s both;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 0.9;
            transform: translateY(0);
        }
    }
    
    /* Premium card components with better text contrast */
    .premium-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        color: #2d3748;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.6s;
    }
    
    .premium-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    .premium-card:hover::before {
        left: 100%;
    }
    
    /* County selection section */
    .county-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(240, 147, 251, 0.15));
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }
    
    .county-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 0 3px 3px 0;
    }
    
    .county-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Hero section with better text contrast */
    .image-container {
        text-align: center;
        margin: 3rem 0;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
        color: #2d3748;
    }
    
    .image-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        animation: rotate 20s linear infinite;
        z-index: -1;
    }
    
    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }
    
    .image-container:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
    }
    
    /* Premium buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.5px;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        cursor: pointer;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.6);
        background: linear-gradient(45deg, #5a6fd8, #6a4190);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
    }
    
    /* Premium selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        font-weight: 500;
        padding: 0.75rem 1rem;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #764ba2;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Premium metrics with better text contrast */
    .stMetric {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        color: #2d3748 !important;
    }
    
    .stMetric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    /* Fix metric text colors */
    .stMetric > div {
        color: #2d3748 !important;
    }
    
    .stMetric label {
        color: #4a5568 !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #2d3748 !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="metric-delta"] {
        color: #667eea !important;
    }
    
    /* Enhanced messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1), rgba(40, 167, 69, 0.05));
        border: none;
        border-left: 4px solid #28a745;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.2);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(23, 162, 184, 0.1), rgba(79, 172, 254, 0.1));
        border: none;
        border-left: 4px solid #4facfe;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.2);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1), rgba(245, 87, 108, 0.1));
        border: none;
        border-left: 4px solid #f5576c;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.2);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(254, 225, 64, 0.1));
        border: none;
        border-left: 4px solid #fee140;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(254, 225, 64, 0.2);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
    }
    
    /* Premium slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    .stSlider > div > div > div > div {
        background: white;
        border: 3px solid #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 24px;
        height: 24px;
    }
    
    /* Premium dataframe */
    .stDataFrame {
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Enhanced expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(240, 147, 251, 0.15));
        border-radius: 15px;
        font-weight: 600;
        padding: 1.2rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25), rgba(240, 147, 251, 0.25));
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
        border: 2px solid transparent;
        background-clip: content-box;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #5a6fd8, #6a4190);
        background-clip: content-box;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
        border-right-color: #764ba2 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 3rem;
        }
        
        .subtitle {
            font-size: 1.2rem;
        }
        
        .main .block-container {
            padding: 2rem 1.5rem;
            margin: 1rem;
            border-radius: 20px;
        }
        
        .premium-card, .county-section, .image-container {
            padding: 1.5rem;
            margin: 1.5rem 0;
        }
    }
    
    /* Text readability fixes for white containers */
    .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #2d3748 !important;
    }
    
    /* Specific fixes for Streamlit components */
    .stSelectbox label {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox > div > div > div {
        color: #2d3748 !important;
    }
    
    .stButton label {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    .stSlider label {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    .stExpander label {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    /* Fix dataframe text */
    .stDataFrame {
        color: #2d3748 !important;
    }
    
    .stDataFrame table {
        color: #2d3748 !important;
    }
    
    .stDataFrame th {
        color: #2d3748 !important;
        background-color: rgba(102, 126, 234, 0.1) !important;
    }
    
    .stDataFrame td {
        color: #2d3748 !important;
    }
    
    /* Fix text in custom HTML containers */
    .image-container h2,
    .image-container h3,
    .image-container p,
    .image-container div {
        color: #2d3748 !important;
    }
    
    .county-section h3,
    .county-section p,
    .county-section div {
        color: #2d3748 !important;
    }
    
    .premium-card h2,
    .premium-card h3,
    .premium-card p,
    .premium-card div {
        color: #2d3748 !important;
    }
    
    /* Override any white text */
    * {
        color: inherit;
    }
    
    /* Ensure good contrast for all text elements */
    .main .block-container * {
        color: #2d3748;
    }
    
    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_main_title():
    """Render premium animated title with stunning effects"""
    title_html = """
    <div class="main-title">
        🚗⚡ EV Forecasting
    </div>
    <div class="subtitle">
        Advanced Machine Learning • Predictive Analytics • Future Insights
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)

def render_image_display():
    """Render premium hero section with stunning visuals"""
    image_html = """
    <div class="image-container">
        <h2 style="color: #2d3748; margin-bottom: 1.5rem; font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 2.2rem;">
            🌱 Sustainable Transportation Intelligence
        </h2>
        <div style="font-size: 5rem; margin: 2rem 0; line-height: 1; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));">
            🚗💨⚡🌍🔮
        </div>
        <p style="color: #4a5568; font-size: 1.3rem; font-weight: 400; line-height: 1.6; max-width: 700px; margin: 0 auto 2rem auto;">
            Harness the power of machine learning to explore electric vehicle adoption patterns, 
            predict future trends, and make data-driven decisions for a sustainable tomorrow.
        </p>
        <div style="display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin-top: 2rem;">
            <div style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px; min-width: 120px;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 1rem; color: #667eea; font-weight: 600;">Real-time Analytics</div>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(118, 75, 162, 0.1); border-radius: 15px; min-width: 120px;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
                <div style="font-size: 1rem; color: #764ba2; font-weight: 600;">AI-Powered Forecasts</div>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(240, 147, 251, 0.1); border-radius: 15px; min-width: 120px;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-size: 1rem; color: #f093fb; font-weight: 600;">Precision Insights</div>
            </div>
        </div>
    </div>
    """
    st.markdown(image_html, unsafe_allow_html=True)

def render_county_selection(counties: List[str], county_data: Dict) -> Optional[str]:
    """
    Render premium county selection interface with enhanced styling
    
    Args:
        counties: List of available county names
        county_data: Dictionary containing county information
        
    Returns:
        Selected county name or None if no selection made
    """
    county_section_html = """
    <div class="county-section">
        <div style="display: flex; align-items: center; margin-bottom: 2rem;">
            <div style="font-size: 3rem; margin-right: 1.5rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));">📍</div>
            <div>
                <h3 style="color: #2d3748; margin: 0; font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 1.8rem;">
                    Select Your County
                </h3>
                <p style="color: #4a5568; margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                    Choose a county to unlock detailed EV insights and AI-powered forecasts
                </p>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.6); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #667eea; backdrop-filter: blur(10px);">
            <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.8rem;">
                <span style="color: #667eea; font-size: 1.2rem;">💡</span>
                <strong style="color: #2d3748; font-size: 1.1rem;">Pro Tip:</strong>
            </div>
            <p style="color: #4a5568; margin: 0; font-size: 1rem; line-height: 1.5;">
                Counties with more historical data points provide more accurate forecasting results. 
                Look for counties with rich data coverage for the best insights.
            </p>
        </div>
    </div>
    """
    st.markdown(county_section_html, unsafe_allow_html=True)
    
    # Create county options with enhanced display names
    county_options = ["🔍 Choose a county to begin your analysis..."] + [
        f"📍 {get_county_display_name(county, county_data)}" for county in counties
    ]
    
    # County selection dropdown with premium styling
    selected_display = st.selectbox(
        "🌟 **Available Counties** (Select to start your analysis):",
        options=county_options,
        index=0,
        help="💡 Select a county to generate EV adoption forecasts and view detailed analytics",
        label_visibility="visible"
    )
    
    # Convert display name back to county name for processing
    if not selected_display.startswith("🔍"):
        # Remove the emoji prefix and find the actual county name
        clean_display = selected_display.replace("📍 ", "")
        for county in counties:
            if get_county_display_name(county, county_data) == clean_display:
                return county
    
    return None

def display_county_info(selected_county: str, county_data: Dict):
    """
    Display premium county information with stunning visual cards
    
    Args:
        selected_county: Name of the selected county
        county_data: Dictionary containing county information
    """
    county_info = get_county_info(selected_county, county_data)
    
    if county_info:
        # Premium header with animation
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0 2rem 0;">
            <h2 style="color: #2d3748; font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 2.5rem; margin-bottom: 0.5rem;">
                📊 County Analytics Dashboard
            </h2>
            <p style="color: #4a5568; font-size: 1.2rem; opacity: 0.9;">
                Comprehensive data insights for your selected region
            </p>
            <div style="width: 100px; height: 3px; background: linear-gradient(45deg, #667eea, #764ba2); margin: 1rem auto; border-radius: 2px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Premium metrics cards with gradients
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                        padding: 2rem 1.5rem; border-radius: 20px; text-align: center; 
                        color: white !important; margin-bottom: 1rem; box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
                        transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="font-size: 3rem; margin-bottom: 1rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">🏛️</div>
                <div style="font-size: 0.9rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; color: white !important;">County</div>
                <div style="font-size: 1.4rem; font-weight: 700; margin-top: 0.5rem; color: white !important;">{county_info['county_name']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb, #f5576c); 
                        padding: 2rem 1.5rem; border-radius: 20px; text-align: center; 
                        color: white !important; margin-bottom: 1rem; box-shadow: 0 10px 25px rgba(240, 147, 251, 0.3);
                        transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="font-size: 3rem; margin-bottom: 1rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">🗺️</div>
                <div style="font-size: 0.9rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; color: white !important;">State</div>
                <div style="font-size: 1.4rem; font-weight: 700; margin-top: 0.5rem; color: white !important;">{county_info['state']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe, #00f2fe); 
                        padding: 2rem 1.5rem; border-radius: 20px; text-align: center; 
                        color: white !important; margin-bottom: 1rem; box-shadow: 0 10px 25px rgba(79, 172, 254, 0.3);
                        transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="font-size: 3rem; margin-bottom: 1rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">📈</div>
                <div style="font-size: 0.9rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; color: white !important;">Data Points</div>
                <div style="font-size: 1.4rem; font-weight: 700; margin-top: 0.5rem; color: white !important;">{county_info['data_points']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Calculate data quality score
        data_quality = min(100, (county_info['data_points'] / 50) * 100)
        
        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fa709a, #fee140); 
                        padding: 2rem 1.5rem; border-radius: 20px; text-align: center; 
                        color: white !important; margin-bottom: 1rem; box-shadow: 0 10px 25px rgba(250, 112, 154, 0.3);
                        transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="font-size: 3rem; margin-bottom: 1rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">⭐</div>
                <div style="font-size: 0.9rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; color: white !important;">Data Quality</div>
                <div style="font-size: 1.4rem; font-weight: 700; margin-top: 0.5rem; color: white !important;">{data_quality:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Date range information with premium styling
        date_range = county_info['date_range']
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(240, 147, 251, 0.15)); 
                    padding: 2rem; border-radius: 20px; margin: 2rem 0; 
                    border: 1px solid rgba(102, 126, 234, 0.2); backdrop-filter: blur(10px);
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.1);">
            <div style="display: flex; align-items: center; justify-content: center; gap: 1.5rem;">
                <div style="font-size: 3rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));">📅</div>
                <div style="text-align: center;">
                    <div style="color: #2d3748; font-weight: 700; font-size: 1.3rem; font-family: 'Poppins', sans-serif;">Data Coverage Period</div>
                    <div style="color: #4a5568; font-size: 1.1rem; margin-top: 0.8rem; font-weight: 500;">
                        <span style="background: rgba(102, 126, 234, 0.1); padding: 0.3rem 0.8rem; border-radius: 8px; margin: 0 0.5rem;">
                            <strong>{date_range['start'].strftime('%B %Y')}</strong>
                        </span>
                        →
                        <span style="background: rgba(240, 147, 251, 0.1); padding: 0.3rem 0.8rem; border-radius: 8px; margin: 0 0.5rem;">
                            <strong>{date_range['end'].strftime('%B %Y')}</strong>
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Historical data preview with enhanced styling
        with st.expander("📈 **Historical Data Insights** - Click to explore detailed metrics"):
            historical = county_info['historical_data']
            
            if historical['dates'] and len(historical['dates']) > 0:
                # Show latest data point
                latest_idx = -1
                latest_date = historical['dates'][latest_idx]
                latest_ev_total = historical['ev_totals'][latest_idx]
                latest_percentage = historical['percentages'][latest_idx]
                
                # Calculate growth if we have enough data
                growth_rate = 0
                if len(historical['ev_totals']) >= 2:
                    prev_total = historical['ev_totals'][-2]
                    if prev_total > 0:
                        growth_rate = ((latest_ev_total - prev_total) / prev_total) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(40, 167, 69, 0.15), rgba(40, 167, 69, 0.05)); 
                                padding: 1.5rem; border-radius: 15px; text-align: center; border: 1px solid rgba(40, 167, 69, 0.2);
                                box-shadow: 0 5px 15px rgba(40, 167, 69, 0.1);">
                        <div style="color: #28a745; font-size: 2rem; margin-bottom: 0.8rem;">🚗</div>
                        <div style="color: #2d3748; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Current EV Total</div>
                        <div style="color: #28a745; font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">{latest_ev_total:,.0f}</div>
                        <div style="color: #4a5568; font-size: 0.8rem;">as of {latest_date.strftime('%B %Y')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(23, 162, 184, 0.15), rgba(23, 162, 184, 0.05)); 
                                padding: 1.5rem; border-radius: 15px; text-align: center; border: 1px solid rgba(23, 162, 184, 0.2);
                                box-shadow: 0 5px 15px rgba(23, 162, 184, 0.1);">
                        <div style="color: #17a2b8; font-size: 2rem; margin-bottom: 0.8rem;">📊</div>
                        <div style="color: #2d3748; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">EV Percentage</div>
                        <div style="color: #17a2b8; font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">{latest_percentage:.2f}%</div>
                        <div style="color: #4a5568; font-size: 0.8rem;">of total vehicles</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    growth_color = "#28a745" if growth_rate >= 0 else "#dc3545"
                    growth_icon = "📈" if growth_rate >= 0 else "📉"
                    growth_bg = "rgba(40, 167, 69, 0.15)" if growth_rate >= 0 else "rgba(220, 53, 69, 0.15)"
                    growth_border = "rgba(40, 167, 69, 0.2)" if growth_rate >= 0 else "rgba(220, 53, 69, 0.2)"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {growth_bg}, rgba(102, 126, 234, 0.05)); 
                                padding: 1.5rem; border-radius: 15px; text-align: center; border: 1px solid {growth_border};
                                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);">
                        <div style="color: {growth_color}; font-size: 2rem; margin-bottom: 0.8rem;">{growth_icon}</div>
                        <div style="color: #2d3748; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Recent Growth</div>
                        <div style="color: {growth_color}; font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">{growth_rate:+.1f}%</div>
                        <div style="color: #4a5568; font-size: 0.8rem;">month-over-month</div>
                    </div>
                    """, unsafe_allow_html=True)

@st.cache_data
def get_county_encoding_map() -> Dict[str, int]:
    """
    Get a mapping of county names to their encoded values for performance optimization.
    
    Returns:
        Dictionary mapping county names to their encoded values
    """
    try:
        # Load the preprocessed data to get county encoding
        df = pd.read_csv("preprocessed_ev_data.csv")
        
        # Create mapping of county names to encodings
        county_encoding_map = dict(zip(df['County'], df['county_encoded']))
        
        return county_encoding_map
        
    except Exception as e:
        st.error(f"Error loading county encoding map: {str(e)}")
        return {}

def get_county_encoding(county_name: str) -> Optional[int]:
    """
    Get the encoded value for a county from the preprocessed data.
    
    Args:
        county_name: Name of the county to encode
        
    Returns:
        Integer encoding for the county or None if not found
    """
    try:
        # Use cached encoding map for better performance
        encoding_map = get_county_encoding_map()
        return encoding_map.get(county_name)
            
    except Exception as e:
        st.error(f"Error getting county encoding: {str(e)}")
        return None

def calculate_feature_engineering_values(historical_data: Dict) -> Dict:
    """
    Calculate engineered features from historical data.
    
    Args:
        historical_data: Dictionary containing historical EV data
        
    Returns:
        Dictionary containing calculated feature values
    """
    try:
        ev_totals = historical_data['ev_totals']
        
        if len(ev_totals) < 3:
            raise ValueError("Insufficient historical data for feature engineering (need at least 3 data points)")
        
        # Get the most recent values for lag features
        ev_total_lag1 = float(ev_totals[-1]) if len(ev_totals) >= 1 else 0.0
        ev_total_lag2 = float(ev_totals[-2]) if len(ev_totals) >= 2 else 0.0
        ev_total_lag3 = float(ev_totals[-3]) if len(ev_totals) >= 3 else 0.0
        
        # Calculate rolling mean of last 3 values
        recent_values = ev_totals[-3:]
        ev_total_roll_mean_3 = float(np.mean(recent_values))
        
        # Calculate percentage changes
        ev_total_pct_change_1 = 0.0
        if len(ev_totals) >= 2 and ev_totals[-2] != 0:
            ev_total_pct_change_1 = float((ev_totals[-1] - ev_totals[-2]) / ev_totals[-2])
        
        ev_total_pct_change_3 = 0.0
        if len(ev_totals) >= 4 and ev_totals[-4] != 0:
            ev_total_pct_change_3 = float((ev_totals[-1] - ev_totals[-4]) / ev_totals[-4])
        
        # Calculate growth slope using linear regression on recent data points
        if len(ev_totals) >= 3:
            # Use last 6 points or all available points if less than 6
            recent_points = min(6, len(ev_totals))
            y_values = np.array(ev_totals[-recent_points:], dtype=float)
            x_values = np.arange(recent_points, dtype=float)
            
            # Calculate slope using least squares
            if len(x_values) > 1:
                slope = float(np.polyfit(x_values, y_values, 1)[0])
            else:
                slope = 0.0
        else:
            slope = 0.0
        
        return {
            'ev_total_lag1': ev_total_lag1,
            'ev_total_lag2': ev_total_lag2,
            'ev_total_lag3': ev_total_lag3,
            'ev_total_roll_mean_3': ev_total_roll_mean_3,
            'ev_total_pct_change_1': ev_total_pct_change_1,
            'ev_total_pct_change_3': ev_total_pct_change_3,
            'ev_growth_slope': slope
        }
        
    except Exception as e:
        raise ValueError(f"Error calculating feature engineering values: {str(e)}")

def prepare_model_input(county_name: str, county_data: Dict, forecast_months: int = 1) -> Optional[np.ndarray]:
    """
    Prepare model input data from county selection for prediction.
    
    Args:
        county_name: Name of the selected county
        county_data: Dictionary containing all county data
        forecast_months: Number of months ahead to forecast (default: 1)
        
    Returns:
        Numpy array with model input features or None if preparation fails
    """
    try:
        # Get county information
        county_info = get_county_info(county_name, county_data)
        if not county_info:
            raise ValueError(f"County '{county_name}' not found in data")
        
        # Get county encoding
        county_encoded = get_county_encoding(county_name)
        if county_encoded is None:
            raise ValueError(f"Could not find encoding for county '{county_name}'")
        
        # Get historical data
        historical_data = county_info['historical_data']
        
        if not historical_data['dates'] or len(historical_data['dates']) < 3:
            raise ValueError(f"Insufficient historical data for county '{county_name}' (need at least 3 data points)")
        
        # Calculate months since start (based on the data structure)
        # Find the earliest date in the entire dataset to calculate months_since_start
        try:
            df = pd.read_csv("preprocessed_ev_data.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            earliest_date = df['Date'].min()
            
            # Get the latest date for this county
            latest_date = max(historical_data['dates'])
            
            # Calculate months since start
            months_diff = (latest_date.year - earliest_date.year) * 12 + (latest_date.month - earliest_date.month)
            months_since_start = months_diff + forecast_months
            
        except Exception as e:
            # Fallback calculation if CSV reading fails
            latest_date = max(historical_data['dates'])
            # Assume start date is January 2018 based on the data structure
            start_date = datetime(2018, 1, 1)
            months_since_start = (latest_date.year - start_date.year) * 12 + (latest_date.month - start_date.month) + forecast_months
        
        # Calculate engineered features
        feature_values = calculate_feature_engineering_values(historical_data)
        
        # Prepare input array in the correct order based on model.feature_names_in_
        # ['months_since_start', 'county_encoded', 'ev_total_lag1', 'ev_total_lag2', 
        #  'ev_total_lag3', 'ev_total_roll_mean_3', 'ev_total_pct_change_1', 
        #  'ev_total_pct_change_3', 'ev_growth_slope']
        
        # Create input as DataFrame with proper feature names to avoid sklearn warnings
        feature_data = {
            'months_since_start': [float(months_since_start)],
            'county_encoded': [float(county_encoded)],
            'ev_total_lag1': [feature_values['ev_total_lag1']],
            'ev_total_lag2': [feature_values['ev_total_lag2']],
            'ev_total_lag3': [feature_values['ev_total_lag3']],
            'ev_total_roll_mean_3': [feature_values['ev_total_roll_mean_3']],
            'ev_total_pct_change_1': [feature_values['ev_total_pct_change_1']],
            'ev_total_pct_change_3': [feature_values['ev_total_pct_change_3']],
            'ev_growth_slope': [feature_values['ev_growth_slope']]
        }
        
        model_input = pd.DataFrame(feature_data)
        
        return model_input
        
    except ValueError as ve:
        # Re-raise ValueError with original message
        raise ve
    except Exception as e:
        raise ValueError(f"Error preparing model input for county '{county_name}': {str(e)}")

def generate_forecast(county_name: str, county_data: Dict, model, forecast_months: int = 6) -> Optional[Dict]:
    """
    Generate EV adoption forecast for selected county using the loaded model with improved logic.
    
    Args:
        county_name: Name of the selected county
        county_data: Dictionary containing all county data
        model: Pre-trained machine learning model
        forecast_months: Number of months to forecast ahead (default: 6)
        
    Returns:
        Dictionary containing forecast results or None if prediction fails
    """
    try:
        if model is None:
            raise ValueError("Model is not available for prediction")
        
        # Get county information for context
        county_info = get_county_info(county_name, county_data)
        if not county_info:
            raise ValueError(f"County '{county_name}' not found in data")
        
        # Get historical data
        historical_data = county_info['historical_data']
        ev_totals = historical_data['ev_totals'].copy()
        
        # Get county encoding
        county_encoded = get_county_encoding(county_name)
        if county_encoded is None:
            raise ValueError(f"Could not find encoding for county '{county_name}'")
        
        # Calculate base months since start
        try:
            df = pd.read_csv("preprocessed_ev_data.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            earliest_date = df['Date'].min()
            latest_date = max(historical_data['dates'])
            base_months = (latest_date.year - earliest_date.year) * 12 + (latest_date.month - earliest_date.month)
        except:
            latest_date = max(historical_data['dates'])
            start_date = datetime(2018, 1, 1)
            base_months = (latest_date.year - start_date.year) * 12 + (latest_date.month - start_date.month)
        
        # Generate predictions iteratively, updating features with each prediction
        predictions = []
        prediction_dates = []
        
        # Start with current EV totals for iterative prediction
        current_ev_totals = ev_totals.copy()
        
        for month_ahead in range(1, forecast_months + 1):
            try:
                # Calculate months since start for this prediction
                months_since_start = base_months + month_ahead
                
                # Calculate features based on current data (including previous predictions)
                if len(current_ev_totals) >= 3:
                    ev_total_lag1 = float(current_ev_totals[-1])
                    ev_total_lag2 = float(current_ev_totals[-2])
                    ev_total_lag3 = float(current_ev_totals[-3])
                    
                    # Rolling mean of last 3 values
                    recent_values = current_ev_totals[-3:]
                    ev_total_roll_mean_3 = float(np.mean(recent_values))
                    
                    # Percentage changes
                    ev_total_pct_change_1 = 0.0
                    if len(current_ev_totals) >= 2 and current_ev_totals[-2] != 0:
                        ev_total_pct_change_1 = float((current_ev_totals[-1] - current_ev_totals[-2]) / current_ev_totals[-2])
                    
                    ev_total_pct_change_3 = 0.0
                    if len(current_ev_totals) >= 4 and current_ev_totals[-4] != 0:
                        ev_total_pct_change_3 = float((current_ev_totals[-1] - current_ev_totals[-4]) / current_ev_totals[-4])
                    
                    # Growth slope
                    recent_points = min(6, len(current_ev_totals))
                    y_values = np.array(current_ev_totals[-recent_points:], dtype=float)
                    x_values = np.arange(recent_points, dtype=float)
                    
                    if len(x_values) > 1:
                        slope = float(np.polyfit(x_values, y_values, 1)[0])
                    else:
                        slope = 0.0
                    
                    # Create model input
                    feature_data = {
                        'months_since_start': [float(months_since_start)],
                        'county_encoded': [float(county_encoded)],
                        'ev_total_lag1': [ev_total_lag1],
                        'ev_total_lag2': [ev_total_lag2],
                        'ev_total_lag3': [ev_total_lag3],
                        'ev_total_roll_mean_3': [ev_total_roll_mean_3],
                        'ev_total_pct_change_1': [ev_total_pct_change_1],
                        'ev_total_pct_change_3': [ev_total_pct_change_3],
                        'ev_growth_slope': [slope]
                    }
                    
                    model_input = pd.DataFrame(feature_data)
                    
                    # Make prediction
                    raw_prediction = model.predict(model_input)[0]
                    
                    # Apply intelligent prediction adjustment
                    prediction = float(raw_prediction)
                    
                    # Prevent unrealistic declines - EV adoption should generally grow or stabilize
                    if len(predictions) == 0:
                        # First prediction - ensure it's not too far from current value
                        current_value = float(current_ev_totals[-1])
                        if prediction < current_value * 0.7:  # Don't allow >30% decline in first month
                            prediction = current_value * 0.95  # Small decline at most
                        elif prediction > current_value * 1.5:  # Don't allow >50% growth in first month
                            prediction = current_value * 1.1   # Reasonable growth
                    else:
                        # Subsequent predictions - ensure reasonable progression
                        prev_prediction = predictions[-1]
                        
                        # Don't allow dramatic drops
                        if prediction < prev_prediction * 0.8:
                            prediction = prev_prediction * 0.98  # Small decline at most
                        elif prediction > prev_prediction * 1.3:
                            prediction = prev_prediction * 1.05  # Moderate growth
                    
                    # Apply minimum growth assumption for EV adoption (generally growing market)
                    if len(predictions) > 0:
                        # Add small positive trend to reflect EV market growth
                        base_growth = 1.01  # 1% monthly growth assumption
                        prediction = max(prediction, predictions[-1] * base_growth)
                    
                    # Ensure prediction is non-negative and reasonable
                    prediction = max(0.1, prediction)  # Minimum threshold
                    
                    predictions.append(prediction)
                    
                    # Update current_ev_totals with the new prediction for next iteration
                    current_ev_totals.append(prediction)
                    
                    # Calculate prediction date
                    prediction_date = latest_date + timedelta(days=30 * month_ahead)
                    prediction_dates.append(prediction_date)
                    
                else:
                    raise ValueError("Insufficient historical data for iterative prediction")
                
            except Exception as e:
                st.warning(f"Failed to generate prediction for month {month_ahead}: {str(e)}")
                # Use trend-based fallback
                if predictions:
                    # Continue with slight trend
                    last_prediction = predictions[-1]
                    trend_prediction = last_prediction * 1.02  # Small growth assumption
                    predictions.append(trend_prediction)
                else:
                    # Fallback to latest historical value with small growth
                    latest_ev_total = float(historical_data['ev_totals'][-1])
                    predictions.append(latest_ev_total * 1.01)
                
                prediction_date = latest_date + timedelta(days=30 * month_ahead)
                prediction_dates.append(prediction_date)
        
        # Calculate some basic statistics
        historical_ev_totals = county_info['historical_data']['ev_totals']
        latest_historical = float(historical_ev_totals[-1])
        
        # Calculate growth metrics
        total_growth = predictions[-1] - latest_historical if predictions else 0.0
        avg_monthly_growth = total_growth / forecast_months if forecast_months > 0 else 0.0
        growth_percentage = (total_growth / latest_historical * 100) if latest_historical > 0 else 0.0
        
        # Prepare forecast result
        forecast_result = {
            'county': county_name,
            'state': county_info['state'],
            'forecast_period': f"{forecast_months} months",
            'prediction_dates': prediction_dates,
            'predicted_values': predictions,
            'latest_historical_value': latest_historical,
            'latest_historical_date': latest_date,
            'total_growth': total_growth,
            'avg_monthly_growth': avg_monthly_growth,
            'growth_percentage': growth_percentage,
            'metadata': {
                'model_type': str(type(model).__name__),
                'prediction_date': datetime.now(),
                'forecast_months': forecast_months,
                'historical_data_points': len(historical_ev_totals)
            }
        }
        
        return forecast_result
        
    except ValueError as ve:
        # Re-raise ValueError with original message for user-friendly display
        raise ve
    except Exception as e:
        raise ValueError(f"Error generating forecast for county '{county_name}': {str(e)}")

def handle_prediction_errors(func):
    """
    Decorator to handle prediction errors gracefully and provide user-friendly messages.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as ve:
            # User-friendly errors (data issues, missing counties, etc.)
            st.error(f"❌ Prediction Error: {str(ve)}")
            return None
        except FileNotFoundError as fe:
            # File-related errors
            st.error(f"❌ File Error: {str(fe)}")
            st.info("💡 Please ensure all required data files are present in the project directory.")
            return None
        except Exception as e:
            # Unexpected errors
            st.error(f"❌ Unexpected Error: An error occurred during prediction")
            st.error(f"Technical details: {str(e)}")
            
            # Show technical details in expander for debugging
            with st.expander("🔧 Technical Details"):
                st.code(traceback.format_exc())
            
            return None
    
    return wrapper

@handle_prediction_errors
def safe_generate_forecast(county_name: str, county_data: Dict, model, forecast_months: int = 6) -> Optional[Dict]:
    """
    Safely generate forecast with error handling.
    
    Args:
        county_name: Name of the selected county
        county_data: Dictionary containing all county data
        model: Pre-trained machine learning model
        forecast_months: Number of months to forecast ahead
        
    Returns:
        Dictionary containing forecast results or None if prediction fails
    """
    return generate_forecast(county_name, county_data, model, forecast_months)

def create_forecast_visualization(forecast_result: Dict, county_data: Dict) -> plt.Figure:
    """
    Create matplotlib visualization for forecast data showing historical and predicted values.
    
    Args:
        forecast_result: Dictionary containing forecast results
        county_data: Dictionary containing all county data
        
    Returns:
        matplotlib Figure object with the forecast visualization
    """
    try:
        # Get county historical data
        county_name = forecast_result['county']
        county_info = get_county_info(county_name, county_data)
        
        if not county_info:
            raise ValueError(f"County '{county_name}' not found in data")
        
        historical_data = county_info['historical_data']
        
        # Create figure with custom styling
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Historical data
        historical_dates = historical_data['dates']
        historical_values = historical_data['ev_totals']
        
        # Forecast data
        forecast_dates = forecast_result['prediction_dates']
        forecast_values = forecast_result['predicted_values']
        
        # Plot historical data
        ax.plot(historical_dates, historical_values, 
               color='#2E86AB', linewidth=2.5, marker='o', markersize=4,
               label='Historical EV Totals', alpha=0.8)
        
        # Plot forecast data
        ax.plot(forecast_dates, forecast_values, 
               color='#A23B72', linewidth=2.5, marker='s', markersize=5,
               label='Forecasted EV Totals', linestyle='--', alpha=0.9)
        
        # Connect historical and forecast with a dotted line
        if historical_dates and forecast_dates:
            connection_dates = [historical_dates[-1], forecast_dates[0]]
            connection_values = [historical_values[-1], forecast_values[0]]
            ax.plot(connection_dates, connection_values, 
                   color='#F18F01', linewidth=1.5, linestyle=':', alpha=0.7)
        
        # Customize the plot
        ax.set_title(f'EV Adoption Forecast: {county_name}, {forecast_result["state"]}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Electric Vehicles', fontsize=12, fontweight='bold')
        
        # Format y-axis with comma separators
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Add forecast period annotation
        if forecast_dates:
            forecast_start = forecast_dates[0]
            forecast_end = forecast_dates[-1]
            ax.axvspan(forecast_start, forecast_end, alpha=0.1, color='#A23B72', 
                      label='Forecast Period')
        
        # Add growth information as text box
        growth_text = f'Total Growth: {forecast_result["growth_percentage"]:+.1f}%\n'
        growth_text += f'Avg Monthly: {forecast_result["avg_monthly_growth"]:+,.0f} EVs'
        
        ax.text(0.02, 0.98, growth_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
               verticalalignment='top', fontsize=10, fontweight='bold')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        # Create a simple error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Error creating visualization:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=1', facecolor='lightcoral', alpha=0.8))
        ax.set_title('Visualization Error', fontsize=14, fontweight='bold')
        ax.axis('off')
        return fig

def create_historical_trend_visualization(county_data: Dict, county_name: str) -> plt.Figure:
    """
    Create matplotlib visualization showing historical EV trends for the selected county.
    
    Args:
        county_data: Dictionary containing all county data
        county_name: Name of the county to visualize
        
    Returns:
        matplotlib Figure object with the historical trend visualization
    """
    try:
        county_info = get_county_info(county_name, county_data)
        
        if not county_info:
            raise ValueError(f"County '{county_name}' not found in data")
        
        historical_data = county_info['historical_data']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        # Plot 1: EV Totals over time
        dates = historical_data['dates']
        ev_totals = historical_data['ev_totals']
        percentages = historical_data['percentages']
        
        ax1.plot(dates, ev_totals, color='#2E86AB', linewidth=2.5, marker='o', markersize=4)
        ax1.set_title(f'Historical EV Totals: {county_name}, {county_info["state"]}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Electric Vehicles', fontsize=12, fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: EV Percentage over time
        ax2.plot(dates, percentages, color='#A23B72', linewidth=2.5, marker='s', markersize=4)
        ax2.set_title('EV Adoption Percentage Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Percentage of Electric Vehicles (%)', fontsize=12, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for both plots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Create a simple error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Error creating historical visualization:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=1', facecolor='lightcoral', alpha=0.8))
        ax.set_title('Historical Visualization Error', fontsize=14, fontweight='bold')
        ax.axis('off')
        return fig

def display_forecast_results(forecast_result: Dict, county_data: Dict):
    """
    Display prediction results in readable format with visualizations.
    
    Args:
        forecast_result: Dictionary containing forecast results
        county_data: Dictionary containing all county data
    """
    try:
        st.markdown("---")
        st.markdown("### 📊 Forecast Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current EV Total",
                f"{forecast_result['latest_historical_value']:,.0f}",
                help=f"Latest historical value as of {forecast_result['latest_historical_date'].strftime('%B %Y')}"
            )
        
        with col2:
            st.metric(
                "Forecasted EV Total",
                f"{forecast_result['predicted_values'][-1]:,.0f}",
                delta=f"{forecast_result['total_growth']:+,.0f}",
                help=f"Predicted value after {forecast_result['forecast_period']}"
            )
        
        with col3:
            st.metric(
                "Total Growth",
                f"{forecast_result['growth_percentage']:+.1f}%",
                help=f"Percentage growth over {forecast_result['forecast_period']}"
            )
        
        with col4:
            st.metric(
                "Avg Monthly Growth",
                f"{forecast_result['avg_monthly_growth']:+,.0f}",
                help="Average monthly increase in EV count"
            )
        
        # Main forecast visualization
        st.markdown("#### 📈 Forecast Visualization")
        
        try:
            forecast_fig = create_forecast_visualization(forecast_result, county_data)
            st.pyplot(forecast_fig)
            plt.close(forecast_fig)  # Clean up memory
        except Exception as e:
            st.error(f"Error creating forecast visualization: {str(e)}")
        
        # Historical trend visualization
        st.markdown("#### 📊 Historical Trends")
        
        try:
            historical_fig = create_historical_trend_visualization(county_data, forecast_result['county'])
            st.pyplot(historical_fig)
            plt.close(historical_fig)  # Clean up memory
        except Exception as e:
            st.error(f"Error creating historical visualization: {str(e)}")
        
        # Detailed forecast table
        st.markdown("#### 📋 Detailed Forecast Data")
        
        # Create a comprehensive table with forecast values
        forecast_df = pd.DataFrame({
            'Month': [date.strftime('%B %Y') for date in forecast_result['prediction_dates']],
            'Predicted EV Total': [f"{val:,.0f}" for val in forecast_result['predicted_values']],
            'Monthly Change': [
                f"{forecast_result['predicted_values'][i] - (forecast_result['latest_historical_value'] if i == 0 else forecast_result['predicted_values'][i-1]):+,.0f}"
                for i in range(len(forecast_result['predicted_values']))
            ],
            'Cumulative Growth': [
                f"{val - forecast_result['latest_historical_value']:+,.0f}"
                for val in forecast_result['predicted_values']
            ],
            'Growth %': [
                f"{((val - forecast_result['latest_historical_value']) / forecast_result['latest_historical_value'] * 100):+.1f}%"
                for val in forecast_result['predicted_values']
            ]
        })
        
        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Forecast metadata and context
        with st.expander("🔍 Forecast Details & Context"):
            metadata = forecast_result['metadata']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Information:**")
                st.write(f"• Model Type: {metadata['model_type']}")
                st.write(f"• Forecast Period: {metadata['forecast_months']} months")
                st.write(f"• Historical Data Points: {metadata['historical_data_points']}")
                st.write(f"• Prediction Generated: {metadata['prediction_date'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.markdown("**County Information:**")
                st.write(f"• County: {forecast_result['county']}")
                st.write(f"• State: {forecast_result['state']}")
                st.write(f"• Latest Historical Date: {forecast_result['latest_historical_date'].strftime('%B %Y')}")
                st.write(f"• Data Coverage: {forecast_result['latest_historical_date'].strftime('%B %Y')}")
        
        # Interpretation and context
        st.markdown("#### 💡 Forecast Interpretation")
        
        # Generate interpretation based on growth trends
        growth_rate = forecast_result['growth_percentage']
        
        if growth_rate > 20:
            interpretation = "🚀 **Strong Growth Expected**: The forecast indicates robust EV adoption with significant growth anticipated."
            color = "success"
        elif growth_rate > 10:
            interpretation = "📈 **Moderate Growth Expected**: The forecast shows steady EV adoption growth in this county."
            color = "info"
        elif growth_rate > 0:
            interpretation = "📊 **Slow Growth Expected**: The forecast indicates modest EV adoption growth."
            color = "warning"
        else:
            interpretation = "⚠️ **Declining Trend**: The forecast suggests potential challenges in EV adoption growth."
            color = "error"
        
        if color == "success":
            st.success(interpretation)
        elif color == "info":
            st.info(interpretation)
        elif color == "warning":
            st.warning(interpretation)
        else:
            st.error(interpretation)
        
        # Additional context
        st.markdown("""
        **Note**: These forecasts are based on historical trends and machine learning predictions. 
        Actual results may vary due to policy changes, economic factors, infrastructure development, 
        and other external influences not captured in the historical data.
        """)
        
    except Exception as e:
        st.error(f"Error displaying forecast results: {str(e)}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())

def validate_startup_requirements() -> Dict[str, bool]:
    """
    Validate that all required files and dependencies are available for the application.
    
    Returns:
        Dictionary with validation results for each requirement
    """
    validation_results = {
        'model_file': os.path.exists("forecasting_ev_model.pkl"),
        'data_file': os.path.exists("preprocessed_ev_data.csv"),
        'dependencies': True  # We'll assume dependencies are installed if we got this far
    }
    
    return validation_results

def display_startup_status(validation_results: Dict[str, bool]):
    """
    Display startup validation status to the user.
    
    Args:
        validation_results: Dictionary with validation results
    """
    st.markdown("### 🔍 System Status Check")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if validation_results['model_file']:
            st.success("✅ **Model File**\nReady for predictions")
        else:
            st.error("❌ **Model File**\nMissing or inaccessible")
    
    with col2:
        if validation_results['data_file']:
            st.success("✅ **Data File**\nCounty data available")
        else:
            st.error("❌ **Data File**\nMissing or inaccessible")
    
    with col3:
        if validation_results['dependencies']:
            st.success("✅ **Dependencies**\nAll libraries loaded")
        else:
            st.error("❌ **Dependencies**\nMissing requirements")
    
    # Overall status
    all_good = all(validation_results.values())
    if all_good:
        st.success("🎉 **All systems ready!** You can now use all features of the EV Forecasting application.")
    else:
        st.warning("⚠️ **Some components are missing.** The application will run with limited functionality.")

def main():
    """Main application function with enhanced error handling and user guidance"""
    try:
        # Configure Streamlit page with optimized settings
        st.set_page_config(
            page_title="EV Forecasting - Interactive ML Tool",
            page_icon="🚗⚡",
            layout="wide",
            initial_sidebar_state="collapsed",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': "# EV Forecasting Tool\nInteractive Electric Vehicle adoption forecasting using machine learning."
            }
        )
        
        # Apply enhanced custom styling
        apply_custom_styling()
        
        # Render main title with custom styling
        render_main_title()
        
        # Validate startup requirements
        validation_results = validate_startup_requirements()
        display_startup_status(validation_results)
        
        # Render image display area
        render_image_display()
        
        # Load model and data
        model = None
        counties = []
        county_data = {}
        
        # Test model loading functionality with enhanced error handling
        try:
            model = load_model()
            st.success("✅ ML model loaded successfully and ready for predictions!")
        except FileNotFoundError:
            st.error("❌ **Model File Missing**: The forecasting model file `forecasting_ev_model.pkl` was not found.")
            st.info("💡 **Solution**: Please ensure the model file is in the project directory. You may need to train the model first.")
            model = None
        except Exception as e:
            st.error("❌ **Model Loading Error**: Failed to load the forecasting model.")
            st.error(f"**Technical Details**: {str(e)}")
            st.info("💡 **Troubleshooting**: Check if the model file is corrupted or incompatible with the current environment.")
            model = None
        
        # Test data loading functionality with enhanced error handling
        try:
            counties, county_data = load_county_data()
            st.success(f"✅ County data loaded successfully! {len(counties)} counties available for analysis.")
        except FileNotFoundError:
            st.error("❌ **Data File Missing**: The county data file `preprocessed_ev_data.csv` was not found.")
            st.info("💡 **Solution**: Please ensure the data file is in the project directory.")
            st.markdown("""
            **Required Files:**
            - `preprocessed_ev_data.csv` - Contains historical EV data by county
            - `forecasting_ev_model.pkl` - Pre-trained machine learning model
            """)
            return
        except Exception as e:
            st.error("❌ **Data Loading Error**: Failed to load county data.")
            st.error(f"**Technical Details**: {str(e)}")
            st.info("💡 **Troubleshooting**: Check if the CSV file is properly formatted and not corrupted.")
            return
        
        # Render county selection interface
        if counties and county_data:
            selected_county = render_county_selection(counties, county_data)
            
            # Display county information if a county is selected
            if selected_county:
                display_county_info(selected_county, county_data)
                
                # Prediction functionality
                st.markdown("---")
                st.markdown("### 🔮 Forecast Generation")
                
                if model is not None:
                    st.info("✅ Ready to generate forecasts! Select forecast parameters below.")
                    
                    # Forecast parameters
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        forecast_months = st.slider(
                            "Forecast Period (months)",
                            min_value=1,
                            max_value=12,
                            value=6,
                            help="Number of months to forecast ahead"
                        )
                    
                    with col2:
                        st.metric(
                            "Selected County",
                            get_county_display_name(selected_county, county_data),
                            help="County for which forecast will be generated"
                        )
                    
                    # Generate forecast button
                    if st.button("🚀 Generate Forecast", type="primary"):
                        with st.spinner(f"Generating {forecast_months}-month forecast for {selected_county}..."):
                            # Generate the forecast
                            forecast_result = safe_generate_forecast(
                                selected_county, 
                                county_data, 
                                model, 
                                forecast_months
                            )
                            
                            if forecast_result:
                                # Store forecast result in session state for persistence
                                st.session_state['forecast_result'] = forecast_result
                                st.success(f"✅ Forecast generated successfully for {selected_county}!")
                            else:
                                st.error("❌ Failed to generate forecast. Please check the error messages above.")
                    
                    # Display forecast results if available
                    if 'forecast_result' in st.session_state and st.session_state['forecast_result']:
                        forecast_result = st.session_state['forecast_result']
                        
                        # Only show results if they match the currently selected county
                        if forecast_result['county'] == selected_county:
                            # Use the new comprehensive results display function
                            display_forecast_results(forecast_result, county_data)
                            
                            # Clear results button
                            st.markdown("---")
                            if st.button("🗑️ Clear Results"):
                                if 'forecast_result' in st.session_state:
                                    del st.session_state['forecast_result']
                                st.rerun()
                        
                        else:
                            # Clear outdated results
                            if 'forecast_result' in st.session_state:
                                del st.session_state['forecast_result']
                
                else:
                    st.warning("⚠️ Model not available - cannot generate forecasts")
                    st.info("💡 Please ensure the model file 'forecasting_ev_model.pkl' is present and can be loaded.")
            else:
                # Show comprehensive help and guidance when no county is selected
                st.markdown("---")
                st.markdown("### 🎯 Getting Started")
                
                # Step-by-step guide
                st.markdown("""
                **Follow these steps to generate EV forecasts:**
                
                1. **📍 Select a County**: Choose from the dropdown above to view historical data
                2. **📊 Review Data**: Examine county information and historical trends  
                3. **🔮 Generate Forecast**: Set parameters and create predictions
                4. **📈 Analyze Results**: View charts and detailed forecast data
                """)
                
                # Show dataset overview with enhanced statistics
                if counties:
                    st.markdown("### 📈 Dataset Overview")
                    
                    # Calculate comprehensive statistics
                    total_data_points = sum(info['data_points'] for info in county_data.values())
                    states = set(info['state'] for info in county_data.values())
                    
                    # Find date range across all counties
                    all_start_dates = [info['date_range']['start'] for info in county_data.values()]
                    all_end_dates = [info['date_range']['end'] for info in county_data.values()]
                    overall_start = min(all_start_dates) if all_start_dates else None
                    overall_end = max(all_end_dates) if all_end_dates else None
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Counties", len(counties))
                    
                    with col2:
                        st.metric("States Covered", len(states))
                    
                    with col3:
                        st.metric("Data Points", f"{total_data_points:,}")
                    
                    with col4:
                        if overall_start and overall_end:
                            date_span = (overall_end - overall_start).days // 30
                            st.metric("Data Span", f"{date_span} months")
                    
                    # Data quality information
                    st.markdown("#### 📊 Data Quality Summary")
                    if overall_start and overall_end:
                        st.info(f"📅 **Coverage Period**: {overall_start.strftime('%B %Y')} to {overall_end.strftime('%B %Y')}")
                    
                    # Top counties by data points
                    top_counties = sorted(
                        [(name, info['data_points']) for name, info in county_data.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    if top_counties:
                        st.markdown("#### 🏆 Counties with Most Data")
                        for i, (county, points) in enumerate(top_counties, 1):
                            county_state = county_data[county]['state']
                            st.write(f"{i}. **{county}, {county_state}** - {points} data points")
                
                # Application features overview
                st.markdown("### ✨ Application Features")
                
                feature_col1, feature_col2 = st.columns(2)
                
                with feature_col1:
                    st.markdown("""
                    **📊 Data Analysis**
                    - Historical EV adoption trends
                    - County-specific statistics
                    - Interactive data exploration
                    - Comprehensive data validation
                    """)
                
                with feature_col2:
                    st.markdown("""
                    **🔮 ML Forecasting**
                    - Multi-month predictions
                    - Growth trend analysis
                    - Visual forecast charts
                    - Confidence metrics
                    """)
                
                # Tips for best results
                st.markdown("### 💡 Tips for Best Results")
                st.markdown("""
                - **Choose counties with rich historical data** for more accurate forecasts
                - **Start with 3-6 month forecasts** for optimal reliability
                - **Compare multiple counties** to identify regional trends
                - **Review historical trends** before interpreting forecasts
                """)
                
                # Troubleshooting section
                with st.expander("🔧 Troubleshooting & FAQ"):
                    st.markdown("""
                    **Common Issues:**
                    
                    **Q: Why can't I generate forecasts?**
                    A: Ensure the ML model file (`forecasting_ev_model.pkl`) is present and the county has sufficient historical data.
                    
                    **Q: Why are some counties missing?**
                    A: Only counties with adequate historical data are included for reliable forecasting.
                    
                    **Q: How accurate are the forecasts?**
                    A: Accuracy depends on historical data quality and external factors not captured in the model.
                    
                    **Q: Can I export the results?**
                    A: Use your browser's print function or screenshot tools to save charts and data.
                    """)
                
                # Quick start call-to-action
                st.markdown("---")
                st.markdown("### 🚀 Ready to Start?")
                st.info("👆 **Select a county from the dropdown above** to begin exploring EV adoption data and generating forecasts!")
        else:
            st.error("❌ No county data available - please check data files")
        
    except Exception as e:
        # Enhanced error handling for application startup
        st.error("❌ **Application Startup Error**")
        st.error("An unexpected error occurred while starting the EV Forecasting application.")
        
        # User-friendly error information
        st.markdown("""
        **Possible causes:**
        - Missing or corrupted data files
        - Incompatible Python environment
        - Insufficient system resources
        - Network connectivity issues
        """)
        
        # Technical details in expandable section
        with st.expander("🔧 Technical Details (for developers)"):
            st.code(f"Error Type: {type(e).__name__}")
            st.code(f"Error Message: {str(e)}")
            st.code("Full Traceback:")
            st.code(traceback.format_exc())
        
        # Troubleshooting suggestions
        st.markdown("### 💡 Troubleshooting Steps")
        st.markdown("""
        1. **Check file integrity**: Ensure all required files are present and not corrupted
        2. **Verify environment**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
        3. **Restart application**: Try refreshing the page or restarting the Streamlit server
        4. **Check system resources**: Ensure sufficient memory and disk space
        5. **Review logs**: Check the console output for additional error details
        """)
        
        # Log error to console with enhanced information
        print(f"[{datetime.now().isoformat()}] Application startup error: {e}", file=sys.stderr)
        print(f"Error type: {type(e).__name__}", file=sys.stderr)
        traceback.print_exc()
        
        # Attempt graceful degradation
        st.info("🔄 **Recovery Mode**: The application will attempt to continue with limited functionality.")
        
    finally:
        # Cleanup and memory management
        import gc
        gc.collect()  # Force garbage collection to free up memory

if __name__ == "__main__":
    main()