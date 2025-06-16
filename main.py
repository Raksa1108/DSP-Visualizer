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
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import page modules
from idk import home, dft, fft_analysis, filtering, convolution, modulation, spectral_analysis, transforms

# Beautiful Ocean Theme CSS
st.markdown("""
<style>
    /* Import beautiful fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #f8fdff 0%, #e8f8ff 50%, #f0fffe 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #e8f4f8 0%, #d4edda 100%);
        border-right: 2px solid rgba(79, 172, 254, 0.1);
    }
    
    .css-1d391kg .css-1vq4p4l {
        color: #2c5282;
        font-weight: 500;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(79, 172, 254, 0.2);
        font-family: 'Inter', sans-serif;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%);
        border: 2px solid rgba(79, 172, 254, 0.2);
        border-radius: 15px;
        color: #2c5282;
        font-weight: 500;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: #4facfe;
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .stSlider > div > div > div > div {
        background: #4facfe;
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(79, 172, 254, 0.3);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background: rgba(232, 244, 248, 0.5);
        border: 2px solid rgba(79, 172, 254, 0.2);
        border-radius: 10px;
        color: #2c5282;
        font-weight: 500;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4facfe;
        box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(232, 244, 248, 0.5);
        border: 2px solid rgba(79, 172, 254, 0.2);
        border-radius: 10px;
        color: #2c5282;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4facfe;
        box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
    }
    
    /* Metrics styling */
    .css-1xarl3l {
        background: linear-gradient(135deg, rgba(232, 244, 248, 0.6) 0%, rgba(212, 237, 218, 0.6) 100%);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(79, 172, 254, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%);
        border-radius: 10px;
        border: 1px solid rgba(79, 172, 254, 0.2);
        color: #2c5282;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 0 0 10px 10px;
        border: 1px solid rgba(79, 172, 254, 0.1);
        border-top: none;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%);
        border-radius: 15px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        color: #2c5282;
        font-weight: 500;
        border: 1px solid rgba(79, 172, 254, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-color: transparent;
    }
    
    /* Info boxes */
    .stAlert {
        background: linear-gradient(135deg, rgba(232, 244, 248, 0.8) 0%, rgba(212, 237, 218, 0.8) 100%);
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 12px;
        color: #2c5282;
    }
    
    /* Sidebar elements */
    .css-1d391kg .stSelectbox label {
        color: #2c5282;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #2c5282;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Plot container */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.1);
        background: white;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%);
        border: 2px dashed rgba(79, 172, 254, 0.3);
        border-radius: 15px;
        color: #2c5282;
    }
    
    /* Data frame */
    .dataframe {
        border-radius: 10px;
        border: 1px solid rgba(79, 172, 254, 0.2);
        background: white;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(232, 244, 248, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    }
    
    /* Loading spinner */
    .stSpinner {
        color: #4facfe;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, rgba(212, 237, 218, 0.8) 0%, rgba(200, 230, 201, 0.8) 100%);
        border-left: 4px solid #4caf50;
        border-radius: 8px;
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, rgba(255, 235, 238, 0.8) 0%, rgba(255, 205, 210, 0.8) 100%);
        border-left: 4px solid #f44336;
        border-radius: 8px;
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 248, 225, 0.8) 0%, rgba(255, 236, 179, 0.8) 100%);
        border-left: 4px solid #ff9800;
        border-radius: 8px;
    }
    
    /* Code block */
    .stCode {
        background: rgba(232, 244, 248, 0.3);
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 8px;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, rgba(232, 244, 248, 0.9) 0%, rgba(212, 237, 218, 0.9) 100%);
        color: #2c5282;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(79, 172, 254, 0.1);
    }
    
    /* Animate on load */
    .main-title {
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Hover effects for containers */
    .main .block-container:hover {
        box-shadow: 0 12px 40px rgba(79, 172, 254, 0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Beautiful animated title
    st.markdown('<h1 class="main-title">🌊 Digital Signal Processing Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation with ocean theme
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Home", "📊 DFT Analysis", "⚡ FFT Analysis", "🔄 Transforms", "🔍 Filtering", "🌊 Convolution", "📡 Modulation", "📈 Spectral Analysis"],
        format_func=lambda x: x
    )
    
    # Add some visual separation
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Explore the depths of signal processing*")
    
    # Display selected page
    page_key = page.split(" ", 1)[1]  # Remove emoji for function calls
    
    if page_key == "Home":
        home.show()
    elif page_key == "DFT Analysis":
        dft.show()
    elif page_key == "FFT Analysis":
        fft_analysis.show()
    elif page_key == "Transforms":
        transforms.show()
    elif page_key == "Filtering":
        filtering.show()
    elif page_key == "Convolution":
        convolution.show()
    elif page_key == "Modulation":
        modulation.show()
    elif page_key == "Spectral Analysis":
        spectral_analysis.show()
    
    # Beautiful footer
    st.markdown("""
    <div class="footer">
        🌊 Made with love for Digital Signal Processing • Powered by Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
