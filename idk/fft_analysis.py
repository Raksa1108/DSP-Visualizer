import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.fft
import scipy.signal
from scipy.signal import windows  # Explicitly import windows module for compatibility
from scipy.io import wavfile
import io

def show():
    """Display FFT analysis page."""
    
    st.markdown("## 📊 Fast Fourier Transform (FFT) Analysis")
    st.markdown("Analyze the frequency components of signals using FFT.")
    
    # Signal source selection
    signal_source = st.radio("Select Signal Source:", 
                            ["Use Sample Signal from Home", "Generate New Signal", "Upload Audio File"])
    
    if signal_source == "Use Sample Signal from Home":
        if 'sample_signal' not in st.session_state:
            st.warning("Please generate a sample signal from the Home tab first!")
            return
        
        signal = st.session_state.sample_signal
        t = st.session_state.sample_time
        fs = st.session_state.sample_fs
        
    elif signal_source == "Generate New Signal":
        st.markdown("### Signal Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            freq1 = st.slider("Frequency 1 (Hz)", 1, 200, 50)
            amp1 = st.slider("Amplitude 1", 0.1, 2.0, 1.0)
        
        with col2:
            freq2 = st.slider("Frequency 2 (Hz)", 1, 200, 120)
            amp2 = st.slider("Amplitude 2", 0.1, 2.0, 0.5)
        
        with col3:
            freq3 = st.slider("Frequency 3 (Hz)", 1, 200, 80)
            amp3 = st.slider("Amplitude 3", 0.1, 2.0, 0.3)
        
        duration = st.slider("Duration (seconds)", 0.5, 5.0, 2.0)
        fs = st.selectbox("Sampling Rate (Hz)", [1000, 2000, 4000, 8000], index=2)
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.05)
        
        # Generate multi-frequency signal
        t = np.linspace(0, duration, int(fs * duration))
        signal = (amp1 * np.sin(2 * np.pi * freq1 * t) + 
                 amp2 * np.sin(2 * np.pi * freq2 * t) + 
                 amp3 * np.sin(2 * np.pi * freq3 * t))
        
        # Add noise
        noise = noise_level * np.random.randn(len(t))
        signal = signal + noise
    
    else:  # Upload Audio File
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
        
        if uploaded_file is not None:
            try:
                # Read audio file
                fs, signal = wavfile.read(uploaded_file)
                
                # Convert to mono if stereo
                if len(signal.shape) > 1:
                    signal = np.mean(signal, axis=1)
                
                # Normalize
                signal = signal.astype(float) / np.max(np.abs(signal))
                
                # Create time vector
                t = np.linspace(0, len(signal)/fs, len(signal))
                
                # Limit duration for performance
                max_duration = 10  # seconds
                if len(signal) > max_duration * fs:
                    signal = signal[:int(max_duration * fs)]
                    t = t[:int(max_duration * fs)]
                    st.info(f"Audio truncated to {max_duration} seconds for performance.")
                
            except Exception as e:
                st.error(f"Error reading audio file: {e}")
                return
        else:
            st.info("Please upload an audio file to proceed.")
            return
    
    # FFT Analysis Options
    st.markdown("### FFT Analysis Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window_type = st.selectbox("Window Function", 
                                 ["None", "Hamming", "Hanning", "Blackman", "Kaiser"])
    
    with col2:
        zero_padding = st.selectbox("Zero Padding", ["None", "2x", "4x", "8x"])
    
    with col3:
        db_scale = st.checkbox("Display in dB", value=True)
    
    # Apply window function
    windowed_signal = signal.copy()
    if window_type != "None":
        if window_type == "Hamming":
            window = windows.hamming(len(signal))
        elif window_type == "Hanning":
            window = windows.hann(len(signal))
        elif window_type == "Blackman":
            window = windows.blackman(len(signal))
        elif window_type == "Kaiser":
            beta = st.slider("Kaiser Beta", 0.1, 10.0, 5.0)
            window = windows.kaiser(len(signal), beta)
        
        windowed_signal = signal * window
    
    # Apply zero padding and adjust time vector
    original_length = len(windowed_signal)
    if zero_padding != "None":
        padding_factor = int(zero_padding[0])
        zeros_to_add = len(windowed_signal) * (padding_factor - 1)
        windowed_signal = np.pad(windowed_signal, (0, zeros_to_add), 'constant')
        # Extend time vector for padded signal
        total_samples = len(windowed_signal)
        duration_extended = total_samples / fs
        t_extended = np.linspace(0, duration_extended, total_samples)
    else:
        t_extended = t
    
    # Compute FFT
    fft_result = scipy.fft.fft(windowed_signal)
    freqs = scipy.fft.fftfreq(len(windowed_signal), 1/fs)
    
    # Only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    magnitude = np.abs(fft_result[:len(fft_result)//2])
    phase = np.angle(fft_result[:len(fft_result)//2])
    
    # Convert to dB if requested
    if db_scale:
        magnitude_db = 20 * np.log10(magnitude + 1e-12)  # Add small value to avoid log(0)
        magnitude_display = magnitude_db
        magnitude_label = "Magnitude (dB)"
    else:
        magnitude_display = magnitude
        magnitude_label = "Magnitude"
    
    # Create visualizations
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Original Signal', 'Windowed Signal', 
                       'FFT Magnitude', 'FFT Phase',
                       'Magnitude Spectrum (Zoomed)', 'Phase Spectrum (Zoomed)'],
        specs=[[{}, {}], [{}, {}], [{}, {}]],
        vertical_spacing=0.08
    )
    
    # Original signal
    fig.add_trace(
        go.Scatter(x=t, y=signal, mode='lines', name='Original', 
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    # Windowed signal (use t_extended for proper time axis after padding)
    fig.add_trace(
        go.Scatter(x=t_extended, y=windowed_signal, 
                  mode='lines', name='Windowed', line=dict(color='red')),
        row=1, col=2
    )
    
    # FFT Magnitude
    fig.add_trace(
        go.Scatter(x=positive_freqs, y=magnitude_display, 
                  mode='lines', name='Magnitude', line=dict(color='green')),
        row=2, col=1
    )
    
    # FFT Phase
    fig.add_trace(
        go.Scatter(x=positive_freqs, y=phase, 
                  mode='lines', name='Phase', line=dict(color='purple')),
        row=2, col=2
    )
    
    # Zoomed magnitude (up to fs/4)
    zoom_freq = fs / 4
    zoom_mask = positive_freqs <= zoom_freq
    fig.add_trace(
        go.Scatter(x=positive_freqs[zoom_mask], y=magnitude_display[zoom_mask], 
                  mode='lines', name='Magnitude (Zoomed)', line=dict(color='orange')),
        row=3, col=1
    )
    
    # Zoomed phase (up to fs/4)
    fig.add_trace(
        go.Scatter(x=positive_freqs[zoom_mask], y=phase[zoom_mask], 
                  mode='lines', name='Phase (Zoomed)', line=dict(color='brown')),
        row=3, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=2)
    
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=2)
    fig.update_yaxes(title_text=magnitude_label, row=2, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=2, col=2)
    fig.update_yaxes(title_text=magnitude_label, row=3, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=3, col=2)
    
    fig.update_layout(height=1000, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Peak detection and analysis
    st.markdown("### Peak Analysis")
    
    # Find peaks
    peaks, properties = scipy.signal.find_peaks(magnitude, height=np.max(magnitude)*0.1)
    peak_freqs = positive_freqs[peaks]
    peak_magnitudes = magnitude[peaks]
    
    if len(peaks) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Detected Peaks")
            peak_df = {
                'Frequency (Hz)': np.round(peak_freqs, 2),
                'Magnitude': np.round(peak_magnitudes, 4),
                'Magnitude (dB)': np.round(20 * np.log10(peak_magnitudes + 1e-12), 2)
            }
            st.dataframe(peak_df)
        
        with col2:
            st.markdown("#### Statistics")
            st.write(f"Total Peaks Found: {len(peaks)}")
            st.write(f"Dominant Frequency: {peak_freqs[np.argmax(peak_magnitudes)]:.2f} Hz")
            st.write(f"Frequency Resolution: {positive_freqs[1] - positive_freqs[0]:.2f} Hz")
            st.write(f"Nyquist Frequency: {fs/2:.0f} Hz")
    
    # Download option
    if st.button("Download FFT Data"):
        # Create download data
        fft_data = {
            'Frequency (Hz)': positive_freqs,
            'Magnitude': magnitude,
            'Phase (rad)': phase
        }
        
        # Convert to CSV
        import pandas as pd
        df = pd.DataFrame(fft_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="fft_analysis.csv",
            mime="text/csv"
        )
