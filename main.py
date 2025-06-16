import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.signal
import scipy.fft
from scipy.io import wavfile
import io
import base64

# Configure page
st.set_page_config(
    page_title="DSP Visualizer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import page modules
from idk import home, dft, fft_analysis, filtering, convolution, modulation, spectral_analysis, transforms

# Custom CSS for pastel ocean gradient styling
st.markdown("""
    <style>
    /* Gradient background for the whole app */
    body, .stApp {
        background: linear-gradient(135deg, #b6eaff 0%, #a6f9e3 100%) !important;
        background-attachment: fixed !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #c3eafe 0%, #a8f7b4 100%) !important;
        color: #003b46 !important;
        border-right: 2px solid #e0f7fa;
    }

    /* Main container tweaks */
    .main .block-container {
        background: rgba(255,255,255,0.7);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        box-shadow: 0 8px 28px 0 rgba(44, 62, 80, 0.05);
    }

    /* Titles and headers */
    h1, h2, h3, h4, h5, h6 {
        color: #007c91 !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-weight: 600;
    }

    /* Sidebar selectbox tweaks */
    .stSelectbox > div {
        background-color: #e3f6fc !important;
        color: #003b46 !important;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #71e3fa 0%, #7fffd4 100%);
        color: #003b46;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        transition: 0.2s;
        box-shadow: 0 4px 16px 0 rgba(44, 62, 80, 0.07);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #7fffd4 0%, #71e3fa 100%);
        color: #007c91;
    }

    /* Dataframe and charts background */
    .stDataFrame, .stPlotlyChart, .element-container {
        background-color: rgba(255,255,255,0.6) !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Main title
    st.markdown('🎵 <span style="font-size:2.6rem;font-weight:700;">Digital Signal Processing Visualizer</span>', unsafe_allow_html=True)
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "DFT Analysis", "FFT Analysis","Transforms", "Filtering", "Convolution", "Modulation", "Spectral Analysis"]
    )
    
    # Display selected page
    if page == "Home":
        home.show()
    elif page == "DFT Analysis":
        dft.show()
    elif page == "FFT Analysis":
        fft_analysis.show()
    elif page == "Transforms":
        transforms.show()
    elif page == "Filtering":
        filtering.show()
    elif page == "Convolution":
        convolution.show()
    elif page == "Modulation":
        modulation.show()
    elif page == "Spectral Analysis":
        spectral_analysis.show()

if __name__ == "__main__":
    main()
