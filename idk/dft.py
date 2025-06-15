import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.fft
import scipy.signal

def dft(signal):
    """Compute the Discrete Fourier Transform of the input signal."""
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, signal)

def idft(X):
    """Compute the Inverse Discrete Fourier Transform of the input spectrum."""
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return np.dot(e, X) / N

def generate_time_signal(signal_type, freq, amp, samples, sample_rate, noise_level):
    """Generate a time-domain signal based on user parameters."""
    t = np.linspace(0, samples / sample_rate, samples, endpoint=False)
    if signal_type == "Sine":
        signal = amp * np.sin(2 * np.pi * freq * t)
    elif signal_type == "Square":
        signal = amp * scipy.signal.square(2 * np.pi * freq * t)
    elif signal_type == "Triangle":
        signal = amp * scipy.signal.sawtooth(2 * np.pi * freq * t, 0.5)
    noise = np.random.normal(0, noise_level, samples)
    return signal + noise, t

def generate_freq_signal(signal_type, freq, amp, samples, sample_rate):
    """Generate a frequency-domain signal based on user parameters."""
    freqs = np.fft.fftfreq(samples, 1/sample_rate)
    spectrum = np.zeros(samples, dtype=complex)
    idx = np.abs(freqs - freq).argmin()  # Find closest frequency bin
    if signal_type == "Single Tone":
        spectrum[idx] = amp * samples / 2  # Scale for correct IDFT amplitude
        spectrum[-idx] = amp * samples / 2  # Conjugate symmetry for real signal
    elif signal_type == "Bandlimited":
        bandwidth = min(freq * 2, sample_rate / 4)
        idx_band = np.where((np.abs(freqs) >= freq - bandwidth/2) & (np.abs(freqs) <= freq + bandwidth/2))[0]
        spectrum[idx_band] = amp * samples / len(idx_band)
    return spectrum, freqs

def show():
    st.markdown('<h2 class="sub-header">DFT Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for SFT and IDFT
    tab1, tab2 = st.tabs(["SFT (Sum of Frequencies Transform)", "IDFT (Inverse DFT)"])
    
    with tab1:
        st.markdown("### Discrete Fourier Transform (DFT)")
        
        # Create columns for controls and plots
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Signal Parameters")
            signal_type = st.selectbox(
                "Signal Type",
                ["Sine", "Square", "Triangle"],
                key="dft_signal_type"
            )
            freq = st.slider("Signal Frequency (Hz)", 0.5, 50.0, 5.0, 0.5, key="dft_freq")
            amp = st.slider("Signal Amplitude", 0.1, 2.0, 1.0, 0.1, key="dft_amp")
            sample_rate = st.slider("Sample Rate (Hz)", 100, 2000, 1000, 50, key="dft_sample_rate")
            samples = st.slider("Number of Samples", 16, 512, 128, 8, key="dft_samples")
            noise_level = st.slider("Noise Level", 0.0, 0.5, 0.0, 0.01, key="dft_noise")
        
        with col2:
            # Generate signal and compute DFT
            signal, t = generate_time_signal(signal_type, freq, amp, samples, sample_rate, noise_level)
            dft_result = dft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Input Signal", "DFT Magnitude Spectrum"),
                vertical_spacing=0.08
            )
            
            # Plot input signal
            fig.add_trace(
                go.Scatter(x=t, y=signal, name="Input Signal", line=dict(color="blue", width=2)),
                row=1, col=1
            )
            
            # Plot DFT magnitude
            fig.add_trace(
                go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(dft_result[:len(dft_result)//2]), 
                           name="DFT Magnitude", line=dict(color="green", width=2)),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                title_text="DFT Analysis",
                showlegend=True,
                template="plotly_white"
            )
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Magnitude", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Frequency domain analysis
        st.markdown("---")
        st.markdown("### Frequency Domain Analysis")
        freq_col1, freq_col2 = st.columns(2)
        
        with freq_col1:
            fft_result = np.fft.fft(signal)  # Compare with scipy.fft
            fig_freq = go.Figure()
            fig_freq.add_trace(
                go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(fft_result[:len(fft_result)//2]),
                           name="FFT Magnitude (SciPy)", line=dict(color="red", dash="dash"))
            )
            fig_freq.add_trace(
                go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(dft_result[:len(dft_result)//2]),
                           name="DFT Magnitude", line=dict(color="green"))
            )
            fig_freq.update_layout(
                title="DFT vs FFT Magnitude Comparison",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                template="plotly_white"
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        
        with freq_col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("### Signal Analysis")
            signal_power = np.mean(signal**2)
            dft_power = np.mean(np.abs(dft_result)**2) / len(signal)
            snr_db = 10 * np.log10(signal_power / (noise_level**2 + 1e-10))
            st.markdown(f"""
            **Signal Parameters:**
            - Signal Frequency: {freq:.1f} Hz
            - Signal Amplitude: {amp:.1f}
            - Sample Rate: {sample_rate} Hz
            - Number of Samples: {samples}
            - Signal Power: {signal_power:.4f}
            - DFT Power: {dft_power:.4f}
            - SNR: {snr_db:.2f} dB
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Theory section
        st.markdown("---")
        st.markdown("### DFT Theory")
        theory_col1, theory_col2 = st.columns(2)
        
        with theory_col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("#### Discrete Fourier Transform (DFT)")
            st.markdown("""
            - **Equation**: X[k] = Σ x[n] * e^(-j2πkn/N)
            - **Purpose**: Converts time-domain signal to frequency domain
            - **Complexity**: O(N²)
            - **Advantages**: Exact frequency analysis
            - **Applications**: Signal processing, spectral analysis
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with theory_col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("#### Fast Fourier Transform (FFT)")
            st.markdown("""
            - **Relation**: Optimized DFT algorithm
            - **Complexity**: O(N log N)
            - **Advantages**: Computationally efficient
            - **Disadvantages**: Requires power-of-2 samples for Cooley-Tukey
            - **Applications**: Real-time signal processing
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced analysis
        st.markdown("---")
        st.markdown("### Advanced Analysis")
        if st.checkbox("Show Phase Spectrum", key="dft_phase"):
            fig_phase = go.Figure()
            fig_phase.add_trace(
                go.Scatter(x=freqs[:len(freqs)//2], y=np.angle(dft_result[:len(dft_result)//2], deg=True),
                           name="DFT Phase", line=dict(color="purple"))
            )
            fig_phase.update_layout(
                title="DFT Phase Spectrum",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Phase (degrees)",
                template="plotly_white"
            )
            st.plotly_chart(fig_phase, use_container_width=True)
    
    with tab2:
        st.markdown("### Inverse Discrete Fourier Transform (IDFT)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Frequency Domain Parameters")
            signal_type = st.selectbox(
                "Spectrum Type",
                ["Single Tone", "Bandlimited"],
                key="idft_signal_type"
            )
            freq = st.slider("Center Frequency (Hz)", 0.5, 50.0, 5.0, 0.5, key="idft_freq")
            amp = st.slider("Spectrum Amplitude", 0.1, 2.0, 1.0, 0.1, key="idft_amp")
            sample_rate = st.slider("Sample Rate (Hz)", 100, 2000, 1000, 50, key="idft_sample_rate")
            samples = st.slider("Number of Samples", 16, 512, 128, 8, key="idft_samples")
        
        with col2:
            # Generate frequency-domain signal and compute IDFT
            freq_signal, freqs = generate_freq_signal(signal_type, freq, amp, samples, sample_rate)
            idft_result = idft(freq_signal)
            t = np.linspace(0, samples / sample_rate, samples, endpoint=False)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Input Frequency Spectrum (Magnitude)", "IDFT Result (Real Part)"),
                vertical_spacing=0.08
            )
            
            # Plot frequency domain
            fig.add_trace(
                go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(freq_signal[:len(freq_signal)//2]),
                           name="Frequency Magnitude", line=dict(color="green", width=2)),
                row=1, col=1
            )
            
            # Plot IDFT result
            fig.add_trace(
                go.Scatter(x=t, y=np.real(idft_result), name="IDFT (Real)", line=dict(color="blue", width=2)),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                title_text="IDFT Analysis",
                showlegend=True,
                template="plotly_white"
            )
            fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Frequency domain analysis
        st.markdown("---")
        st.markdown("### Signal Analysis")
        freq_col1, freq_col2 = st.columns(2)
        
        with freq_col1:
            fft_result = np.fft.ifft(freq_signal)  # Compare with scipy.fft
            fig_comp = go.Figure()
            fig_comp.add_trace(
                go.Scatter(x=t, y=np.real(idft_result), name="IDFT (Real)", line=dict(color="blue"))
            )
            fig_comp.add_trace(
                go.Scatter(x=t, y=np.real(fft_result), name="IFFT (SciPy)", line=dict(color="red", dash="dash"))
            )
            fig_comp.update_layout(
                title="IDFT vs IFFT Comparison",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                template="plotly_white"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with freq_col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("### Signal Analysis")
            signal_power = np.mean(np.abs(idft_result)**2)
            freq_power = np.mean(np.abs(freq_signal)**2) / samples
            st.markdown(f"""
            **Signal Parameters:**
            - Center Frequency: {freq:.1f} Hz
            - Spectrum Amplitude: {amp:.1f}
            - Sample Rate: {sample_rate} Hz
            - Number of Samples: {samples}
            - Time-Domain Power: {signal_power:.4f}
            - Frequency-Domain Power: {freq_power:.4f}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Theory section
        st.markdown("---")
        st.markdown("### IDFT Theory")
        theory_col1, theory_col2 = st.columns(2)
        
        with theory_col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("#### Inverse DFT")
            st.markdown("""
            - **Equation**: x[n] = (1/N) * Σ X[k] * e^(j2πkn/N)
            - **Purpose**: Converts frequency-domain to time-domain
            - **Complexity**: O(N²)
            - **Advantages**: Accurate signal reconstruction
            - **Applications**: Signal synthesis, filtering
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with theory_col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("#### Practical Considerations")
            st.markdown("""
            - **Symmetry**: Real signals require conjugate-symmetric DFT
            - **Scaling**: IDFT includes 1/N normalization
            - **Applications**: Audio processing, image reconstruction
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced analysis
        st.markdown("---")
        st.markdown("### Advanced Analysis")
        if st.checkbox("Show Imaginary Part", key="idft_imag"):
            fig_imag = go.Figure()
            fig_imag.add_trace(
                go.Scatter(x=t, y=np.imag(idft_result), name="IDFT (Imaginary)", line=dict(color="purple"))
            )
            fig_imag.update_layout(
                title="IDFT Imaginary Part",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                template="plotly_white"
            )
            st.plotly_chart(fig_imag, use_container_width=True)

if __name__ == "__main__":
    show()
