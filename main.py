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
from idk import home, fft_analysis, filtering, convolution, modulation, spectral_analysis

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main title
    st.markdown('<h1 class="main-header">🎵 Digital Signal Processing Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "FFT Analysis", "Filtering", "Convolution", "Modulation", "Spectral Analysis"]
    )
    
    # Display selected page
    if page == "Home":
        home.show()
    elif page == "FFT Analysis":
        fft_analysis.show()
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
