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

# Custom CSS for better styling
st.markdown("""

""", unsafe_allow_html=True)

def main():
    # Main title
    st.markdown('🎵 Digital Signal Processing Visualizer', unsafe_allow_html=True)
    
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
