import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal

def show():
    st.markdown("""
    ## 🎵 Welcome to DSP Visualizer!

    This interactive web application is designed to help students, engineers, and enthusiasts explore the fascinating world of **Digital Signal Processing (DSP)** through hands-on visualizations and real-time experimentation. Whether you're learning DSP concepts for the first time or applying advanced techniques, this tool provides an intuitive platform to:

    - **Visualize Signals**: Generate and analyze signals in both time and frequency domains.
    - **Explore Transforms**: Understand Fourier transforms (DTFT, CTFT, CTFS, DTFS) and their properties.
    - **Apply DSP Techniques**: Experiment with filtering, convolution, modulation, and spectral analysis.
    - **Analyze Real Data**: Upload audio files to perform real-time signal processing.
    - **Learn Interactively**: Adjust parameters and see immediate updates to deepen your understanding.

    ### What is Digital Signal Processing?
    DSP involves manipulating digital signals to extract information, enhance quality, or transform them for specific applications. From audio processing in music apps to image enhancement in cameras, DSP is at the heart of modern technology. This app covers key DSP concepts, including:

    - **Time and Frequency Domains**: How signals are represented and transformed.
    - **Transforms**: Converting signals between domains using Fourier series and transforms.
    - **Filtering**: Removing unwanted components from signals.
    - **Convolution**: Combining signals to model system responses.
    - **Modulation**: Encoding information for communication systems.
    - **Spectral Analysis**: Analyzing frequency content of signals.

    ### App Features
    Navigate through the sidebar to explore the following pages:

    - **Home**: Generate and visualize sample signals (you're here!).
    - **DFT Analysis**: Compute and analyze the Discrete Fourier Transform of signals.
    - **FFT Analysis**: Explore the Fast Fourier Transform for efficient frequency analysis.
    - **Transforms**: Dive into Discrete-Time Fourier Transform (DTFT), Continuous-Time Fourier Transform (CTFT), Continuous-Time Fourier Series (CTFS), and Discrete-Time Fourier Series (DTFS).
    - **Filtering**: Design and apply digital filters (low-pass, high-pass, band-pass, band-stop).
    - **Convolution**: Understand how signals interact through convolution operations.
    - **Modulation**: Experiment with Amplitude Modulation (AM), Frequency Modulation (FM), and Phase Modulation (PM).
    - **Spectral Analysis**: Upload audio files for real-time frequency and time-domain analysis.

    ### Getting Started
    1. Use the sidebar to select a page.
    2. Experiment with the signal generator below or upload your own audio files on relevant pages.
    3. Adjust sliders and parameters to see real-time changes in visualizations.
    4. Download processed signals or results where available.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📈 Advanced Signal Generator")
        
        # Signal parameters
        signal_type = st.selectbox("Signal Type", 
                                 ["Sine Wave", "Square Wave", "Sawtooth", "Triangle", "Chirp", "White Noise"],
                                 key="home_signal_type")
        
        duration = st.slider("Duration (seconds)", 0.5, 10.0, 2.0, key="home_duration")
        fs = st.slider("Sampling Rate (Hz)", 500, 48000, 1000, key="home_fs")
        amplitude = st.slider("Amplitude", 0.1, 2.0, 1.0, key="home_amplitude")
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1, key="home_noise_level")
        
        # Additional parameters based on signal type
        if signal_type in ["Sine Wave", "Square Wave", "Sawtooth", "Triangle"]:
            freq = st.slider("Frequency (Hz)", 1, min(fs//2, 1000), 10, key="home_freq")
        elif signal_type == "Chirp":
            f0 = st.slider("Start Frequency (Hz)", 1, min(fs//4, 500), 10, key="home_f0")
            f1 = st.slider("End Frequency (Hz)", 1, min(fs//2, 1000), 100, key="home_f1")
        else:  # White Noise
            freq = None  # No frequency needed
        
        # Generate time vector
        t = np.linspace(0, duration, int(fs * duration))
        
        # Generate signal based on type
        if signal_type == "Sine Wave":
            signal = amplitude * np.sin(2 * np.pi * freq * t)
        elif signal_type == "Square Wave":
            signal = amplitude * np.sign(np.sin(2 * np.pi * freq * t))
        elif signal_type == "Sawtooth":
            signal = amplitude * (2 * (t * freq - np.floor(t * freq + 0.5)))
        elif signal_type == "Triangle":
            signal = amplitude * 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - amplitude
        elif signal_type == "Chirp":
            signal = amplitude * scipy.signal.chirp(t, f0, duration, f1, method='linear')
        else:  # White Noise
            signal = amplitude * np.random.randn(len(t))
        
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
            # Compute FFT for frequency domain analysis
            fft_signal = np.fft.fft(st.session_state.sample_signal)
            freqs = np.fft.fftfreq(len(st.session_state.sample_signal), 1/st.session_state.sample_fs)
            
            # Filter positive frequencies for magnitude and phase plots
            positive_idx = freqs >= 0
            positive_freqs = freqs[positive_idx]
            magnitude = np.abs(fft_signal[positive_idx])
            phase = np.angle(fft_signal[positive_idx])
            
            # Create a 2x2 grid for the four plots
            col1, col2 = st.columns(2)
            
            # Time Domain Plot
            with col1:
                fig_time = go.Figure()
                fig_time.add_trace(
                    go.Scatter(
                        x=st.session_state.sample_time, 
                        y=st.session_state.sample_signal,
                        mode='lines', 
                        name='Signal', 
                        line=dict(color='#1f77b4')
                    )
                )
                fig_time.update_layout(
                    title="Time Domain",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    height=350,
                    margin=dict(l=50, r=20, t=50, b=50),
                    template="plotly_white",
                    showlegend=False
                )
                fig_time.update_xaxes(gridcolor='lightgrey')
                fig_time.update_yaxes(gridcolor='lightgrey')
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Frequency Domain Plot (Real and Imaginary Parts)
            with col2:
                fig_freq = go.Figure()
                fig_freq.add_trace(
                    go.Scatter(
                        x=freqs, 
                        y=np.real(fft_signal),
                        mode='lines', 
                        name='Real', 
                        line=dict(color='#ff7f0e')
                    )
                )
                fig_freq.add_trace(
                    go.Scatter(
                        x=freqs, 
                        y=np.imag(fft_signal),
                        mode='lines', 
                        name='Imaginary', 
                        line=dict(color='#2ca02c')
                    )
                )
                fig_freq.update_layout(
                    title="Frequency Domain",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Amplitude",
                    height=350,
                    margin=dict(l=50, r=20, t=50, b=50),
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig_freq.update_xaxes(gridcolor='lightgrey')
                fig_freq.update_yaxes(gridcolor='lightgrey')
                st.plotly_chart(fig_freq, use_container_width=True)
            
            # Magnitude Spectrum Plot
            with col1:
                fig_mag = go.Figure()
                fig_mag.add_trace(
                    go.Scatter(
                        x=positive_freqs, 
                        y=magnitude,
                        mode='lines', 
                        name='Magnitude', 
                        line=dict(color='#d62728')
                    )
                )
                fig_mag.update_layout(
                    title="Magnitude Spectrum",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Magnitude",
                    height=350,
                    margin=dict(l=50, r=20, t=50, b=50),
                    template="plotly_white",
                    showlegend=False
                )
                fig_mag.update_xaxes(gridcolor='lightgrey')
                fig_mag.update_yaxes(gridcolor='lightgrey')
                st.plotly_chart(fig_mag, use_container_width=True)
            
            # Phase Spectrum Plot
            with col2:
                fig_phase = go.Figure()
                fig_phase.add_trace(
                    go.Scatter(
                        x=positive_freqs, 
                        y=phase,
                        mode='lines', 
                        name='Phase', 
                        line=dict(color='#9467bd')
                    )
                )
                fig_phase.update_layout(
                    title="Phase Spectrum",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Phase (rad)",
                    height=350,
                    margin=dict(l=50, r=20, t=50, b=50),
                    template="plotly_white",
                    showlegend=False
                )
                fig_phase.update_xaxes(gridcolor='lightgrey')
                fig_phase.update_yaxes(gridcolor='lightgrey')
                st.plotly_chart(fig_phase, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📚 DSP Basics and Use Cases

    Digital Signal Processing is a cornerstone of modern technology. Here are some fundamental concepts and their real-world applications:

    - **Time-Domain Analysis**: Understand signal behavior over time, crucial for audio recording and playback.
    - **Frequency-Domain Analysis**: Identify frequency components, used in music equalizers and speech recognition.
    - **Fourier Transforms**: Convert signals between time and frequency domains, essential for image compression (JPEG) and audio compression (MP3).
    - **Filtering**: Remove noise or isolate frequency bands, applied in medical imaging (ECG, MRI) and telecommunications.
    - **Convolution**: Model system responses, used in reverb effects for music production.
    - **Modulation**: Encode signals for transmission, critical for radio, Wi-Fi, and satellite communications.
    - **Spectral Analysis**: Analyze frequency content, used in seismic monitoring and audio forensics.

    This app lets you experiment with these concepts interactively, bridging theory and practice.

    ### 💡 Quick Tips for Using the App

    - **Experiment Freely**: Adjust parameters like frequency, amplitude, and noise to see their impact on signals.
    - **Upload Audio**: Use the Spectral Analysis page to process your own WAV or MP3 files.
    - **Explore Transforms**: The Transforms page offers deep insights into DTFT, CTFT, CTFS, and DTFS with interactive visualizations.
    - **Real-Time Feedback**: All plots update instantly as you change parameters, making it easy to learn cause-and-effect relationships.
    - **Download Results**: Save processed signals or analysis results on pages like Filtering and Spectral Analysis.
    - **Check Sampling Rates**: Ensure the sampling rate (fs) is at least twice the highest frequency (Nyquist theorem) to avoid aliasing.

    ### 🔧 Technical Features

    - **Signal Generation**: Supports multiple signal types (sine, square, sawtooth, triangle, chirp, white noise) with customizable parameters.
    - **Sampling Rates**: Flexible sampling rates from 500 Hz to 48 kHz, suitable for audio and other signals.
    - **Filters**: Implement IIR and FIR filters with types including Butterworth, Chebyshev, and custom designs.
    - **Window Functions**: Apply Hamming, Hanning, Blackman, and Kaiser windows for spectral analysis.
    - **Modulation Techniques**: Simulate AM, FM, and PM with adjustable carrier and modulation frequencies.
    - **Efficient Algorithms**: Leverages NumPy, SciPy, and FFT for fast, interactive processing.
    - **Interactive Visualizations**: Powered by Plotly for dynamic, high-quality plots.

    ### 🌟 Why Use This Visualizer?

    - **Educational**: Perfect for students learning DSP concepts in courses like signals and systems or communications.
    - **Professional**: Useful for engineers prototyping signal processing algorithms.
    - **Hobbyist-Friendly**: Engaging for audio enthusiasts or makers experimenting with sound design.
    - **Cross-Platform**: Runs in any modern web browser via Streamlit, no installation required (except for local development).

    ### 📢 Feedback and Contributions

    Have suggestions or found a bug? Let us know by contacting the development team or contributing to the open-source project. Your feedback helps make this tool better for everyone!
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
