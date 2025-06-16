st.markdown("""
    <style>
    /* Gradient background for the whole app */
    body, .stApp {
        background: linear-gradient(135deg, #e6f7ff 0%, #b3ecf0 50%, #87ceeb 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f8ff 0%, #e0f6f6 100%) !important;
        color: #2c5f5f !important;
        border-right: 2px solid #b3ecf0;
        box-shadow: 2px 0 10px rgba(176, 224, 230, 0.3);
    }
    
    /* Main container tweaks */
    .main .block-container {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        box-shadow: 0 12px 35px rgba(135, 206, 235, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(179, 236, 240, 0.3);
    }
    
    /* Titles and headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2c5f5f !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(135, 206, 235, 0.1);
    }
    
    /* Sidebar selectbox tweaks */
    .stSelectbox > div {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f7ff 100%) !important;
        color: #2c5f5f !important;
        border-radius: 12px;
        border: 1px solid #b3ecf0;
        box-shadow: 0 2px 8px rgba(135, 206, 235, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #b3ecf0 0%, #87ceeb 100%);
        color: #2c5f5f;
        border-radius: 15px;
        border: 1px solid rgba(179, 236, 240, 0.5);
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(135, 206, 235, 0.2);
        backdrop-filter: blur(5px);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #87ceeb 0%, #b3ecf0 100%);
        color: #1a4545;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(135, 206, 235, 0.3);
    }
    
    /* Dataframe and charts background */
    .stDataFrame, .stPlotlyChart, .element-container {
        background: rgba(240, 248, 255, 0.8) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(179, 236, 240, 0.3);
        box-shadow: 0 4px 15px rgba(135, 206, 235, 0.1);
        backdrop-filter: blur(5px);
    }
    
    /* Input fields */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background: rgba(240, 248, 255, 0.9) !important;
        border: 1px solid #b3ecf0 !important;
        border-radius: 10px !important;
        color: #2c5f5f !important;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: rgba(240, 248, 255, 0.7);
        border: 1px solid rgba(179, 236, 240, 0.4);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(135, 206, 235, 0.1);
        backdrop-filter: blur(3px);
    }
    </style>
""", unsafe_allow_html=True)
