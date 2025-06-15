import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show():
    
    st.markdown("""
    ## Welcome to the Digital Signal Processing Visualizer! 🎉
    
    This interactive tool helps you explore various digital signal processing concepts through 
    visualization and hands-on experimentation. You can:
    
    - 📊 **FFT Analysis**: Analyze frequency components of signals
    - 🔧 **Filtering**: Apply various digital filters (Low-pass, High-pass, Band-pass)
    - 🔄 **Convolution**: Understand convolution operations
    - 📡 **Modulation**: Explore AM, FM, and PM modulation techniques
    - 🎛️ **Spectral Analysis**: Real-time analysis of uploaded audio files
    
    ### Getting Started
    1. Navigate between tabs using the menu above
    2. Upload your own audio files or use generated signals
    3. Adjust parameters and see real-time updates
    4. Download processed results
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Sample Signal Generator")
        
        # Signal parameters
        freq = st.slider("Frequency (Hz)", 1, 100, 10)
        amplitude = st.slider("Amplitude", 0.1, 2.0, 1.0)
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1)
        duration = st.slider("Duration (seconds)", 0.5, 5.0, 2.0)
        
        signal_type = st.selectbox("Signal Type", 
                                 ["Sine Wave", "Square Wave", "Sawtooth", "Triangle"])
        
        # Generate sample rate and time vector
        fs = 1000  # Sample rate
        t = np.linspace(0, duration, int(fs * duration))
        
        # Generate signal based on type
        if signal_type == "Sine Wave":
            signal = amplitude * np.sin(2 * np.pi * freq * t)
        elif signal_type == "Square Wave":
            signal = amplitude * np.sign(np.sin(2 * np.pi * freq * t))
        elif signal_type == "Sawtooth":
            signal = amplitude * (2 * (t * freq - np.floor(t * freq + 0.5)))
        else:  # Triangle
            signal = amplitude * 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - amplitude
        
        # Add noise
        noise = noise_level * np.random.randn(len(t))
        noisy_signal = signal + noise
        
        # Store in session state for other pages
        st.session_state.sample_signal = noisy_signal
        st.session_state.sample_time = t
        st.session_state.sample_fs = fs
    
    with col2:
        st.markdown("### 📊 Signal Visualization")
        
        if 'sample_signal' in st.session_state:
            # Time domain plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Time Domain', 'Frequency Domain'],
                vertical_spacing=0.1
            )
            
            # Time domain
            fig.add_trace(
                go.Scatter(x=st.session_state.sample_time, 
                          y=st.session_state.sample_signal,
                          mode='lines', name='Signal', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Frequency domain
            fft_signal = np.fft.fft(st.session_state.sample_signal)
            freqs = np.fft.fftfreq(len(st.session_state.sample_signal), 1/st.session_state.sample_fs)
            
            # Only plot positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            magnitude = np.abs(fft_signal[:len(fft_signal)//2])
            
            fig.add_trace(
                go.Scatter(x=positive_freqs, y=magnitude,
                          mode='lines', name='Magnitude', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Quick tips
    st.markdown("""
    ### 💡 Quick Tips
    
    - **Upload Audio**: Use the Spectral Analysis tab to upload and analyze your own audio files
    - **Real-time Updates**: All visualizations update automatically when you change parameters
    - **Download Results**: Most tabs allow you to download processed signals
    - **Parameter Exploration**: Try different parameter combinations to understand their effects
    
    ### 🔧 Technical Features
    
    - **Sampling Rates**: Support for various sampling rates (8kHz to 48kHz)
    - **Filter Types**: IIR and FIR filters with customizable parameters
    - **Window Functions**: Hamming, Hanning, Blackman, and more
    - **Modulation**: AM, FM, PM with carrier frequency control
    - **Real-time Processing**: Efficient algorithms for interactive exploration
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit and Python scientific libraries</p>
    </div>
    """, unsafe_allow_html=True)
