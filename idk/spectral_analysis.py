import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.signal
from scipy.io import wavfile
import io

def show():
    st.markdown('<h2 class="sub-header">🎛️ Spectral Analysis</h2>', unsafe_allow_html=True)
    
    # Create columns for controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Signal Configuration")
        
        # Signal source selection (removed Real-time Simulation)
        signal_source = st.selectbox(
            "Signal Source",
            ["Generated Signal", "Upload Audio File"]
        )
        
        if signal_source == "Generated Signal":
            # Signal parameters
            duration = st.slider("Duration (seconds)", 0.5, 10.0, 3.0, 0.1)
            sample_rate = st.slider("Sample Rate (Hz)", 1000, 16000, 8000, 500)
            
            # Create time vector and validate
            num_samples = int(sample_rate * duration)
            if num_samples <= 0:
                st.error(f"Invalid number of samples: {num_samples}. Ensure duration and sample rate are positive.")
                return
            t = np.linspace(0, duration, num_samples)
            
            # Debug information
            st.write(f"Debug: Duration = {duration} s, Sample Rate = {sample_rate} Hz, Number of Samples = {len(t)}")
            
            # Signal type
            signal_type = st.selectbox(
                "Signal Type",
                ["Multi-tone", "Chirp", "Noisy Sine", "Square Wave", "Sawtooth", "White Noise", "Pink Noise"]
            )
            
            if signal_type == "Multi-tone":
                st.markdown("#### Frequency Components")
                num_tones = st.slider("Number of Tones", 1, 5, 3)
                frequencies = []
                amplitudes = []
                phases = []
                
                for i in range(num_tones):
                    col_freq, col_amp, col_phase = st.columns(3)
                    with col_freq:
                        freq = st.slider(f"Freq {i+1} (Hz)", 1.0, 1000.0, 50.0 * (i+1), 1.0, key=f"freq_{i}")
                        frequencies.append(freq)
                    with col_amp:
                        amp = st.slider(f"Amp {i+1}", 0.1, 2.0, 1.0, 0.1, key=f"amp_{i}")
                        amplitudes.append(amp)
                    with col_phase:
                        phase = st.slider(f"Phase {i+1} (°)", 0, 360, 0, 15, key=f"phase_{i}")
                        phases.append(np.radians(phase))
                        
            elif signal_type == "Chirp":
                f_start = st.slider("Start Frequency (Hz)", 1.0, 1000.0, 10.0, 1.0)
                f_end = st.slider("End Frequency (Hz)", 1.0, 1000.0, 500.0, 1.0)
                
            elif signal_type == "Noisy Sine":
                signal_freq = st.slider("Signal Frequency (Hz)", 1.0, 500.0, 50.0, 1.0)
                noise_level = st.slider("Noise Level", 0.0, 1.0, 0.2, 0.05)
                
            elif signal_type in ["Square Wave", "Sawtooth"]:
                fundamental_freq = st.slider("Fundamental Frequency (Hz)", 1.0, 200.0, 25.0, 1.0)
                
            # Generate signal based on type
            try:
                if signal_type == "Multi-tone":
                    signal = np.zeros_like(t)
                    for freq, amp, phase in zip(frequencies, amplitudes, phases):
                        signal += amp * np.sin(2 * np.pi * freq * t + phase)
                        
                elif signal_type == "Chirp":
                    signal = scipy.signal.chirp(t, f_start, duration, f_end)
                    
                elif signal_type == "Noisy Sine":
                    signal = np.sin(2 * np.pi * signal_freq * t) + noise_level * np.random.randn(len(t))
                    
                elif signal_type == "Square Wave":
                    signal = scipy.signal.square(2 * np.pi * fundamental_freq * t)
                    
                elif signal_type == "Sawtooth":
                    signal = scipy.signal.sawtooth(2 * np.pi * fundamental_freq * t)
                    
                elif signal_type == "White Noise":
                    signal = np.random.randn(len(t))
                    
                elif signal_type == "Pink Noise":
                    white_noise = np.random.randn(len(t))
                    b, a = scipy.signal.butter(1, 0.1, 'low')
                    signal = scipy.signal.filtfilt(b, a, white_noise)
                
                # Debug signal generation
                st.write(f"Debug: Signal generated for {signal_type}, Length = {len(signal)}")
            
            except Exception as e:
                st.error(f"Error generating signal for {signal_type}: {e}")
                return
        
        elif signal_source == "Upload Audio File":
            uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
            
            if uploaded_file is not None:
                try:
                    # Read audio file
                    sample_rate, signal = wavfile.read(uploaded_file)
                    
                    # Convert to mono if stereo
                    if len(signal.shape) > 1:
                        signal = np.mean(signal, axis=1)
                    
                    # Normalize
                    signal = signal.astype(float) / np.max(np.abs(signal))
                    
                    # Create time vector
                    duration = len(signal) / sample_rate
                    t = np.linspace(0, duration, len(signal))
                    
                    # Limit duration for performance
                    max_duration = 10  # seconds
                    if len(signal) > max_duration * sample_rate:
                        signal = signal[:int(max_duration * sample_rate)]
                        t = t[:int(max_duration * sample_rate)]
                        st.info(f"Audio truncated to {max_duration} seconds for performance.")
                    
                except Exception as e:
                    st.error(f"Error reading audio file: {e}")
                    return
            else:
                st.info("Please upload an audio file to proceed.")
                return
        
        # Analysis parameters
        st.markdown("### Analysis Parameters")
        
        window_type = st.selectbox(
            "Window Function",
            ["Hann", "Hamming", "Blackman", "Kaiser", "Rectangular", "Bartlett"]
        )
        
        if window_type == "Kaiser":
            beta = st.slider("Kaiser Beta", 0.1, 10.0, 5.0, 0.1)
        
        # FFT parameters
        fft_size = st.selectbox("FFT Size", [512, 1024, 2048, 4096, 8192], index=2)
        overlap = st.slider("Overlap (%)", 0, 90, 50, 10)
        
        # Spectral analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Power Spectral Density", "Spectrogram", "Welch's Method", "Periodogram", "Multitaper"]
        )
        
        # Additional options
        show_3d = st.checkbox("3D Visualization")
        log_scale = st.checkbox("Logarithmic Scale", value=True)
    
    with col2:
        # Check if signal is defined before proceeding
        if 'signal' not in locals():
            st.error("Signal not defined. Please ensure a signal source is properly configured.")
            return
        
        # Validate signal length
        if len(signal) == 0:
            st.error("Signal is empty. Please check signal generation parameters.")
            return
        
        # Apply window function
        if window_type == "Hann":
            window = scipy.signal.windows.hann(len(signal))
        elif window_type == "Hamming":
            window = scipy.signal.windows.hamming(len(signal))
        elif window_type == "Blackman":
            window = scipy.signal.windows.blackman(len(signal))
        elif window_type == "Kaiser":
            window = scipy.signal.windows.kaiser(len(signal), beta)
        elif window_type == "Rectangular":
            window = np.ones(len(signal))
        elif window_type == "Bartlett":
            window = scipy.signal.windows.bartlett(len(signal))
        
        windowed_signal = signal * window
        
        # Perform spectral analysis
        if analysis_type == "Power Spectral Density":
            freqs, psd = scipy.signal.welch(signal, sample_rate, nperseg=fft_size, 
                                          noverlap=int(fft_size * overlap / 100))
            
        elif analysis_type == "Welch's Method":
            freqs, psd = scipy.signal.welch(signal, sample_rate, nperseg=fft_size,
                                          noverlap=int(fft_size * overlap / 100))
            
        elif analysis_type == "Periodogram":
            freqs, psd = scipy.signal.periodogram(windowed_signal, sample_rate, nfft=fft_size)
            
        elif analysis_type == "Multitaper":
            freqs, psd = scipy.signal.welch(signal, sample_rate, nperseg=fft_size,
                                          noverlap=int(fft_size * overlap / 100))
            
        # Create main visualization
        if analysis_type == "Spectrogram" or show_3d:
            # Compute spectrogram
            f_spec, t_spec, Sxx = scipy.signal.spectrogram(
                signal, sample_rate, nperseg=fft_size,
                noverlap=int(fft_size * overlap / 100)
            )
            
            if show_3d:
                # 3D Spectrogram
                fig = go.Figure(data=[go.Surface(
                    x=t_spec,
                    y=f_spec,
                    z=10 * np.log10(Sxx + 1e-10) if log_scale else Sxx,
                    colorscale='Viridis'
                )])
                
                fig.update_layout(
                    title='3D Spectrogram',
                    scene=dict(
                        xaxis_title='Time (s)',
                        yaxis_title='Frequency (Hz)',
                        zaxis_title='Power (dB)' if log_scale else 'Power'
                    ),
                    height=600
                )
                
            else:
                # 2D Spectrogram
                fig = go.Figure(data=go.Heatmap(
                    x=t_spec,
                    y=f_spec,
                    z=10 * np.log10(Sxx + 1e-10) if log_scale else Sxx,
                    colorscale='Viridis',
                    colorbar=dict(title='Power (dB)' if log_scale else 'Power')
                ))
                
                fig.update_layout(
                    title='Spectrogram',
                    xaxis_title='Time (s)',
                    yaxis_title='Frequency (Hz)',
                    height=500
                )
        
        else:
            # Create subplots for time and frequency domain
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Time Domain Signal', 'Windowed Signal', 'Power Spectral Density'),
                vertical_spacing=0.08
            )
            
            # Time domain signal
            fig.add_trace(
                go.Scatter(x=t, y=signal, name='Original Signal', 
                          line=dict(color='blue', width=1)),
                row=1, col=1
            )
            
            # Windowed signal
            fig.add_trace(
                go.Scatter(x=t, y=windowed_signal, name='Windowed Signal', 
                          line=dict(color='red', width=1)),
                row=2, col=1
            )
            
            # Add window function (scaled)
            fig.add_trace(
                go.Scatter(x=t, y=window * np.max(signal), name='Window Function', 
                          line=dict(color='green', width=2, dash='dash'),
                          opacity=0.7),
                row=2, col=1
            )
            
            # Power spectral density
            psd_plot = 10 * np.log10(psd + 1e-10) if log_scale else psd
            fig.add_trace(
                go.Scatter(x=freqs, y=psd_plot, name='PSD', 
                          line=dict(color='purple', width=2)),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text=f"Spectral Analysis - {analysis_type}",
                showlegend=True,
                template="plotly_white"
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
            
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            fig.update_yaxes(title_text="Power (dB)" if log_scale else "Power", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis section
    st.markdown("---")
    st.markdown("### 📊 Detailed Spectral Analysis")
    
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        st.markdown("#### Peak Detection")
        
        # Find peaks in the spectrum
        if 'psd' in locals():
            # Find peaks
            peaks, properties = scipy.signal.find_peaks(psd, height=np.max(psd) * 0.1, distance=10)
            peak_freqs = freqs[peaks]
            peak_powers = psd[peaks]
            
            st.markdown("**Detected Peaks:**")
            for i, (freq, power) in enumerate(zip(peak_freqs[:5], peak_powers[:5])):  # Show top 5
                power_db = 10 * np.log10(power + 1e-10)
                st.text(f"Peak {i+1}: {freq:.2f} Hz, {power_db:.2f} dB")
            
            # Spectral features
            st.markdown("#### Spectral Features")
            
            # Calculate spectral centroid
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            
            # Calculate spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
            
            # Calculate spectral rolloff (95% of energy)
            cumulative_energy = np.cumsum(psd)
            total_energy = cumulative_energy[-1]
            rolloff_threshold = 0.95 * total_energy
            rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0][0]
            spectral_rolloff = freqs[rolloff_idx]
            
            st.markdown(f"""
            - **Spectral Centroid**: {spectral_centroid:.2f} Hz
            - **Spectral Bandwidth**: {spectral_bandwidth:.2f} Hz
            - **Spectral Rolloff (95%)**: {spectral_rolloff:.2f} Hz
            - **Total Power**: {np.sum(psd):.6f}
            - **Peak Power**: {np.max(psd):.6f}
            """)
    
    with detail_col2:
        st.markdown("#### Window Function Analysis")
        
        # Window function properties
        window_energy = np.sum(window**2)
        window_power = np.mean(window**2)
        coherent_gain = np.mean(window)
        processing_gain = window_energy / len(window)
        
        st.markdown(f"""
        **Window Properties:**
        - **Type**: {window_type}
        - **Length**: {len(window)} samples
        - **Coherent Gain**: {coherent_gain:.4f}
        - **Processing Gain**: {processing_gain:.4f}
        - **Energy**: {window_energy:.4f}
        - **Power**: {window_power:.4f}
        """)
        
        # Plot window function
        fig_window = go.Figure()
        
        fig_window.add_trace(
            go.Scatter(x=np.arange(len(window)), y=window, 
                      name=f'{window_type} Window',
                      line=dict(color='green', width=2))
        )
        
        fig_window.update_layout(
            title=f"{window_type} Window Function",
            xaxis_title="Sample",
            yaxis_title="Amplitude",
            height=300,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_window, use_container_width=True)
    
    # Advanced analysis options
    st.markdown("---")
    st.markdown("### 🔬 Advanced Analysis")
    
    advanced_col1, advanced_col2 = st.columns(2)
    
    with advanced_col1:
        if st.checkbox("Harmonic Analysis"):
            if signal_source == "Generated Signal" and signal_type in ["Square Wave", "Sawtooth"]:
                # Calculate harmonics
                fundamental = fundamental_freq
                harmonics = []
                harmonic_powers = []
                
                for h in range(1, 11):  # First 10 harmonics
                    harmonic_freq = h * fundamental
                    if harmonic_freq < sample_rate / 2:  # Nyquist limit
                        # Find closest frequency bin
                        freq_idx = np.argmin(np.abs(freqs - harmonic_freq))
                        harmonic_power = psd[freq_idx]
                        harmonics.append(harmonic_freq)
                        harmonic_powers.append(harmonic_power)
                
                # Plot harmonics
                fig_harm = go.Figure()
                
                fig_harm.add_trace(
                    go.Bar(x=harmonics, y=10 * np.log10(np.array(harmonic_powers) + 1e-10),
                          name='Harmonics', marker_color='red')
                )
                
                fig_harm.update_layout(
                    title="Harmonic Content",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Power (dB)",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_harm, use_container_width=True)
                
                # Calculate THD
                fundamental_power = harmonic_powers[0] if harmonic_powers else 0
                total_harmonic_power = np.sum(harmonic_powers[1:]) if len(harmonic_powers) > 1 else 0
                thd = np.sqrt(total_harmonic_power / fundamental_power) * 100 if fundamental_power > 0 else 0
                
                st.markdown(f"**Total Harmonic Distortion (THD)**: {thd:.2f}%")
    
    with advanced_col2:
        if st.checkbox("Frequency Domain Statistics"):
            if 'psd' in locals():
                # Statistical measures of the spectrum
                psd_normalized = psd / np.sum(psd)  # Normalize to probability distribution
                
                # Spectral entropy
                spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))
                
                # Spectral flatness (Wiener entropy)
                geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
                arithmetic_mean = np.mean(psd)
                spectral_flatness = geometric_mean / arithmetic_mean
                
                # Spectral crest factor
                spectral_crest = np.max(psd) / np.mean(psd)
                
                st.markdown(f"""
                **Spectral Statistics:**
                - **Spectral Entropy**: {spectral_entropy:.4f} bits
                - **Spectral Flatness**: {spectral_flatness:.4f}
                - **Spectral Crest Factor**: {spectral_crest:.2f}
                - **Mean Frequency**: {np.mean(freqs):.2f} Hz
                - **Median Frequency**: {np.median(freqs):.2f} Hz
                - **Frequency Range**: {freqs[-1] - freqs[0]:.2f} Hz
                """)
        
        if st.checkbox("Compare Analysis Methods"):
            # Compare different spectral estimation methods
            methods_comparison = {}
            
            # Periodogram
            f_per, psd_per = scipy.signal.periodogram(windowed_signal, sample_rate, nfft=fft_size)
            methods_comparison['Periodogram'] = (f_per, psd_per)
            
            # Welch's method
            f_welch, psd_welch = scipy.signal.welch(signal, sample_rate, nperseg=fft_size//2)
            methods_comparison["Welch's Method"] = (f_welch, psd_welch)
            
            # Plot comparison
            fig_comp = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange']
            for i, (method, (f, psd_method)) in enumerate(methods_comparison.items()):
                psd_db = 10 * np.log10(psd_method + 1e-10)
                fig_comp.add_trace(
                    go.Scatter(x=f, y=psd_db, name=method, 
                              line=dict(color=colors[i % len(colors)], width=2))
                )
            
            fig_comp.update_layout(
                title="Spectral Estimation Methods Comparison",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power (dB)",
                template="plotly_white"
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("Export Spectral Data"):
            if 'freqs' in locals() and 'psd' in locals():
                # Create CSV data
                spectral_data = np.column_stack((freqs, psd, 10 * np.log10(psd + 1e-10)))
                csv_data = "Frequency (Hz),Power,Power (dB)\n"
                for freq, power, power_db in spectral_data:
                    csv_data += f"{freq:.4f},{power:.8f},{power_db:.4f}\n"
                
                st.download_button(
                    label="Download Spectral Data (CSV)",
                    data=csv_data,
                    file_name="spectral_analysis.csv",
                    mime="text/csv"
                )
    
    with export_col2:
        if st.button("Generate Report"):
            report = f"""
# Spectral Analysis Report

## Signal Parameters
- Duration: {duration if 'duration' in locals() else 'N/A'} seconds
- Sample Rate: {sample_rate} Hz
- Signal Type: {signal_type if 'signal_type' in locals() else 'N/A'}
- Window: {window_type}
- FFT Size: {fft_size}

## Analysis Results
- Analysis Method: {analysis_type}
- Frequency Resolution: {sample_rate/fft_size:.4f} Hz
- Number of Frequency Bins: {len(freqs) if 'freqs' in locals() else 'N/A'}

## Spectral Features
- Spectral Centroid: {spectral_centroid:.2f} Hz
- Spectral Bandwidth: {spectral_bandwidth:.2f} Hz
- Spectral Rolloff: {spectral_rolloff:.2f} Hz

Generated on: {st.session_state.get('current_time', 'Unknown')}
            """
            
            st.download_button(
                label="Download Analysis Report",
                data=report,
                file_name="spectral_analysis_report.md",
                mime="text/markdown"
            )
