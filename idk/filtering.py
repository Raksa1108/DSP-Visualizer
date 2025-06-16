import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal
import scipy.fft
from scipy.io import wavfile
import io

def show():
    """Display digital filtering page."""
    
    st.markdown("## 🔧 Digital Filtering")
    st.markdown("Apply various digital filters to signals and analyze their effects.")
    
    # Signal source selection
    signal_source = st.radio("Select Signal Source:", 
                            ["Use Sample Signal from Home", "Generate Noisy Signal", "Upload Audio File"])
    
    if signal_source == "Use Sample Signal from Home":
        if 'sample_signal' not in st.session_state:
            st.warning("Please generate a sample signal from the Home tab first!")
            return
        
        signal = st.session_state.sample_signal
        t = st.session_state.sample_time
        fs = st.session_state.sample_fs
        
    elif signal_source == "Generate Noisy Signal":
        st.markdown("### Signal Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            clean_freq = st.slider("Clean Signal Frequency (Hz)", 1, 100, 20)
            noise_freq = st.slider("Noise Frequency (Hz)", 50, 500, 200)
            
        with col2:
            signal_amp = st.slider("Signal Amplitude", 0.1, 2.0, 1.0)
            noise_amp = st.slider("Noise Amplitude", 0.1, 2.0, 0.5)
        
        duration = st.slider("Duration (seconds)", 0.5, 5.0, 2.0)
        fs = st.selectbox("Sampling Rate (Hz)", [1000, 2000, 4000, 8000], index=2)
        
        # Generate noisy signal
        t = np.linspace(0, duration, int(fs * duration))
        clean_signal = signal_amp * np.sin(2 * np.pi * clean_freq * t)
        noise = noise_amp * np.sin(2 * np.pi * noise_freq * t)
        signal = clean_signal + noise
    
    else:  # Upload Audio File
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
        
        if uploaded_file is not None:
            try:
                fs, signal = wavfile.read(uploaded_file)
                
                if len(signal.shape) > 1:
                    signal = np.mean(signal, axis=1)
                
                signal = signal.astype(float) / np.max(np.abs(signal))
                t = np.linspace(0, len(signal)/fs, len(signal))
                
                # Limit duration for performance
                max_duration = 10
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
    
    # Filter Design Section
    st.markdown("### Filter Design")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.selectbox("Filter Type", 
                                 ["Lowpass", "Highpass", "Bandpass", "Bandstop"])
        
    with col2:
        filter_method = st.selectbox("Filter Method", 
                                   ["Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic"])
    
    with col3:
        filter_order = st.slider("Filter Order", 1, 10, 4)
    
    # Filter parameters based on type
    if filter_type in ["Lowpass", "Highpass"]:
        cutoff_freq = st.slider("Cutoff Frequency (Hz)", 1, fs//2, fs//8)
        critical_freq = cutoff_freq / (fs/2)  # Normalize to Nyquist
        
    else:  # Bandpass or Bandstop
        col1, col2 = st.columns(2)
        with col1:
            low_freq = st.slider("Low Frequency (Hz)", 1, fs//2-1, fs//10)
        with col2:
            high_freq = st.slider("High Frequency (Hz)", low_freq+1, fs//2, fs//4)
        
        critical_freq = [low_freq / (fs/2), high_freq / (fs/2)]
    
    # Additional filter parameters
    if filter_method in ["Chebyshev I", "Elliptic"]:
        rp = st.slider("Passband Ripple (dB)", 0.1, 5.0, 1.0)
    else:
        rp = None
        
    if filter_method in ["Chebyshev II", "Elliptic"]:
        rs = st.slider("Stopband Attenuation (dB)", 10, 80, 40)
    else:
        rs = None
    
    # Design filter
    try:
        if filter_method == "Butterworth":
            sos = scipy.signal.butter(filter_order, critical_freq, 
                                    btype=filter_type.lower(), output='sos')
        elif filter_method == "Chebyshev I":
            sos = scipy.signal.cheby1(filter_order, rp, critical_freq, 
                                    btype=filter_type.lower(), output='sos')
        elif filter_method == "Chebyshev II":
            sos = scipy.signal.cheby2(filter_order, rs, critical_freq, 
                                    btype=filter_type.lower(), output='sos')
        elif filter_method == "Elliptic":
            sos = scipy.signal.ellip(filter_order, rp, rs, critical_freq, 
                                   btype=filter_type.lower(), output='sos')
        
        # Apply filter
        filtered_signal = scipy.signal.sosfilt(sos, signal)
        
    except Exception as e:
        st.error(f"Filter design error: {e}")
        return
    
    # Compute frequency responses
    w, h = scipy.signal.sosfreqz(sos, worN=2048, fs=fs)
    
    # Compute FFTs for comparison
    fft_original = scipy.fft.fft(signal)
    fft_filtered = scipy.fft.fft(filtered_signal)
    freqs = scipy.fft.fftfreq(len(signal), 1/fs)
    
    # Only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    magnitude_original = np.abs(fft_original[:len(fft_original)//2])
    magnitude_filtered = np.abs(fft_filtered[:len(fft_filtered)//2])
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Original Signal', 'Filtered Signal',
                       'Filter Frequency Response', 'Filter Phase Response',
                       'FFT: Original vs Filtered', 'Time Domain Comparison'],
        specs=[[{}, {}], [{}, {}], [{}, {}]],
        vertical_spacing=0.18  # Increased spacing to prevent overlap
    )
    
    # Time domain signals
    fig.add_trace(
        go.Scatter(x=t, y=signal, mode='lines', name='Original', 
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=filtered_signal, mode='lines', name='Filtered', 
                  line=dict(color='red')),
        row=1, col=2
    )
    
    # Filter frequency response
    fig.add_trace(
        go.Scatter(x=w, y=20*np.log10(np.abs(h)), mode='lines', 
                  name='Magnitude Response', line=dict(color='green')),
        row=2, col=1
    )
    
    # Filter phase response
    fig.add_trace(
        go.Scatter(x=w, y=np.angle(h), mode='lines', 
                  name='Phase Response', line=dict(color='purple')),
        row=2, col=2
    )
    
    # FFT comparison
    fig.add_trace(
        go.Scatter(x=positive_freqs, y=20*np.log10(magnitude_original + 1e-12), 
                  mode='lines', name='Original FFT', line=dict(color='blue', dash='solid')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=positive_freqs, y=20*np.log10(magnitude_filtered + 1e-12), 
                  mode='lines', name='Filtered FFT', line=dict(color='red', dash='dash')),
        row=3, col=1
    )
    
    # Time domain overlay
    fig.add_trace(
        go.Scatter(x=t, y=signal, mode='lines', name='Original', 
                  line=dict(color='blue', width=1), opacity=0.7),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=filtered_signal, mode='lines', name='Filtered', 
                  line=dict(color='red', width=2)),
        row=3, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude (dB)", row=2, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=2, col=2)
    fig.update_yaxes(title_text="Magnitude (dB)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=3, col=2)
    
    fig.update_layout(height=1200, showlegend=False)  # Increased height
    st.plotly_chart(fig, use_container_width=True)
    
    # Filter Analysis
    st.markdown("### Filter Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Filter Specifications")
        st.write(f"**Type:** {filter_type} {filter_method}")
        st.write(f"**Order:** {filter_order}")
        if filter_type in ["Lowpass", "Highpass"]:
            st.write(f"**Cutoff:** {cutoff_freq} Hz")
        else:
            st.write(f"**Band:** {low_freq} - {high_freq} Hz")
        
        if rp is not None:
            st.write(f"**Passband Ripple:** {rp} dB")
        if rs is not None:
            st.write(f"**Stopband Attenuation:** {rs} dB")
    
    with col2:
        st.markdown("#### Signal Statistics")
        st.write(f"**Original RMS:** {np.sqrt(np.mean(signal**2)):.4f}")
        st.write(f"**Filtered RMS:** {np.sqrt(np.mean(filtered_signal**2)):.4f}")
        st.write(f"**SNR Improvement:** {10*np.log10(np.var(filtered_signal)/np.var(signal-filtered_signal)):.2f} dB")
        st.write(f"**Signal Length:** {len(signal)} samples")
        st.write(f"**Duration:** {len(signal)/fs:.2f} seconds")
    
    with col3:
        st.markdown("#### Filter Performance")
        # Calculate -3dB bandwidth
        magnitude_response = np.abs(h)
        max_response = np.max(magnitude_response)
        half_power = max_response / np.sqrt(2)
        
        # Find -3dB points
        half_power_indices = np.where(magnitude_response >= half_power)[0]
        if len(half_power_indices) > 0:
            bandwidth = w[half_power_indices[-1]] - w[half_power_indices[0]]
            st.write(f"**-3dB Bandwidth:** {bandwidth:.2f} Hz")
        
        # Convert SOS to transfer function coefficients for group delay
        try:
            b, a = scipy.signal.sos2tf(sos)
            _, gd = scipy.signal.group_delay((b, a), w=w, fs=fs)
            avg_group_delay = np.mean(gd[~np.isnan(gd)])
            st.write(f"**Avg Group Delay:** {avg_group_delay:.4f} s")
        except Exception as e:
            st.warning(f"Group delay calculation failed: {e}")
            avg_group_delay = None
        
        # Filter stability (check poles)
        st.write(f"**Filter Stable:** {'Yes' if np.all(np.abs(np.roots(a)) < 1) else 'No'}")
    
    # Real-time filter parameters adjustment
    st.markdown("### Real-time Parameter Adjustment")
    
    if st.checkbox("Enable Real-time Mode"):
        st.info("Adjust parameters above to see real-time updates!")
        
        # Zero-phase filtering option
        zero_phase = st.checkbox("Zero-phase Filtering (filtfilt)")
        if zero_phase:
            filtered_signal_zp = scipy.signal.sosfiltfilt(sos, signal)
            
            # Add zero-phase filtered signal to comparison
            fig_realtime = go.Figure()
            fig_realtime.add_trace(go.Scatter(x=t, y=signal, mode='lines', 
                                            name='Original', line=dict(color='blue')))
            fig_realtime.add_trace(go.Scatter(x=t, y=filtered_signal, mode='lines', 
                                            name='Causal Filter', line=dict(color='red')))
            fig_realtime.add_trace(go.Scatter(x=t, y=filtered_signal_zp, mode='lines', 
                                            name='Zero-phase Filter', line=dict(color='green')))
            
            fig_realtime.update_layout(title="Filter Comparison: Causal vs Zero-phase",
                                     xaxis_title="Time (s)", yaxis_title="Amplitude",
                                     height=400)
            st.plotly_chart(fig_realtime, use_container_width=True)
    
    # Export options
    st.markdown("### Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Filtered Signal"):
            # Create WAV file
            filtered_int = (filtered_signal * 32767).astype(np.int16)
            wav_io = io.BytesIO()
            wavfile.write(wav_io, fs, filtered_int)
            wav_io.seek(0)
            
            st.download_button(
                label="Download as WAV",
                data=wav_io.getvalue(),
                file_name="filtered_signal.wav",
                mime="audio/wav"
            )
    
    with col2:
        if st.button("Download Filter Coefficients"):
            # Save filter coefficients
            import pandas as pd
            
            # Convert SOS to coefficients
            b, a = scipy.signal.sos2tf(sos)
            filter_data = {
                'Numerator (b)': b,
                'Denominator (a)': np.pad(a, (0, len(b)-len(a)), 'constant')
            }
            
            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in filter_data.items()]))
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="filter_coefficients.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("Download Analysis Data"):
            # Comprehensive analysis data
            import pandas as pd
            
            analysis_data = {
                'Time (s)': t,
                'Original Signal': signal,
                'Filtered Signal': filtered_signal,
                'Frequency (Hz)': w,
                'Filter Magnitude (dB)': 20*np.log10(np.abs(h)),
                'Filter Phase (rad)': np.angle(h)
            }
            
            # Handle different lengths
            max_len = max(len(t), len(w))
            for key, value in analysis_data.items():
                if len(value) < max_len:
                    analysis_data[key] = np.pad(value, (0, max_len - len(value)), 'constant', constant_values=np.nan)
            
            df = pd.DataFrame(analysis_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="filter_analysis.csv",
                mime="text/csv"
            )
