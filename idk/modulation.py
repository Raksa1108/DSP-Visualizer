import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal

def show():
    st.markdown('<h2 class="sub-header">Modulation Techniques</h2>', unsafe_allow_html=True)
    
    # Create columns for controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Modulation Parameters")
        
        # Modulation type (QAM removed)
        modulation_type = st.selectbox(
            "Modulation Type",
            ["Amplitude Modulation (AM)", "Frequency Modulation (FM)", 
             "Phase Modulation (PM)", "Double Sideband (DSB)", 
             "Single Sideband (SSB)"]
        )
        
        # Time parameters
        duration = st.slider("Duration (seconds)", 0.5, 5.0, 2.0, 0.1)
        sample_rate = st.slider("Sample Rate (Hz)", 1000, 10000, 5000, 500)
        
        # Message signal parameters
        st.markdown("#### Message Signal")
        message_freq = st.slider("Message Frequency (Hz)", 0.5, 50.0, 5.0, 0.5)
        message_amp = st.slider("Message Amplitude", 0.1, 2.0, 1.0, 0.1)
        
        # Carrier signal parameters
        st.markdown("#### Carrier Signal")
        carrier_freq = st.slider("Carrier Frequency (Hz)", 50.0, 500.0, 100.0, 10.0)
        carrier_amp = st.slider("Carrier Amplitude", 0.1, 2.0, 1.0, 0.1)
        
        # Modulation index/depth
        if modulation_type in ["Amplitude Modulation (AM)", "Double Sideband (DSB)"]:
            mod_index = st.slider("Modulation Index", 0.1, 2.0, 0.5, 0.1)
        elif modulation_type in ["Frequency Modulation (FM)", "Phase Modulation (PM)"]:
            mod_index = st.slider("Modulation Index", 0.1, 10.0, 2.0, 0.1)
        else:
            mod_index = st.slider("Modulation Parameter", 0.1, 2.0, 1.0, 0.1)
    
    with col2:
        # Create time vector
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Generate message signal
        message_signal = message_amp * np.sin(2 * np.pi * message_freq * t)
        
        # Generate carrier signal
        carrier_signal = carrier_amp * np.sin(2 * np.pi * carrier_freq * t)
        
        # Generate modulated signal based on type
        if modulation_type == "Amplitude Modulation (AM)":
            modulated_signal = (1 + mod_index * message_signal) * carrier_signal
            demod_signal = np.abs(scipy.signal.hilbert(modulated_signal))
            
        elif modulation_type == "Double Sideband (DSB)":
            modulated_signal = mod_index * message_signal * carrier_signal
            demod_signal = modulated_signal * carrier_signal
            window_size = max(1, int(sample_rate / (10 * carrier_freq)))
            demod_signal = np.convolve(demod_signal, np.ones(window_size)/window_size, mode='same')
            
        elif modulation_type == "Frequency Modulation (FM)":
            inst_freq = carrier_freq + mod_index * message_freq * message_signal
            phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
            modulated_signal = carrier_amp * np.sin(phase)
            analytic_signal = scipy.signal.hilbert(modulated_signal)
            demod_signal = np.diff(np.unwrap(np.angle(analytic_signal)))
            demod_signal = np.append(demod_signal, demod_signal[-1])
            
        elif modulation_type == "Phase Modulation (PM)":
            phase = 2 * np.pi * carrier_freq * t + mod_index * message_signal
            modulated_signal = carrier_amp * np.sin(phase)
            analytic_signal = scipy.signal.hilbert(modulated_signal)
            demod_signal = np.diff(np.unwrap(np.angle(analytic_signal)))
            demod_signal = np.append(demod_signal, demod_signal[-1])
            
        elif modulation_type == "Single Sideband (SSB)":
            dsb_signal = message_signal * carrier_signal
            analytic_signal = scipy.signal.hilbert(message_signal)
            modulated_signal = np.real(analytic_signal * np.exp(1j * 2 * np.pi * carrier_freq * t))
            demod_signal = modulated_signal * carrier_signal
            window_size = max(1, int(sample_rate / (10 * carrier_freq)))
            demod_signal = np.convolve(demod_signal, np.ones(window_size)/window_size, mode='same')
        
        # Create subplots with increased spacing and margin to avoid overlap
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Message Signal', 'Carrier Signal', 
                            'Modulated Signal', 'Demodulated Signal'),
            vertical_spacing=0.13  # Increased spacing
        )
        
        # Plot message signal
        fig.add_trace(
            go.Scatter(x=t, y=message_signal, name='Message', 
                       line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Plot carrier signal
        fig.add_trace(
            go.Scatter(x=t[:1000], y=carrier_signal[:1000], name='Carrier', 
                       line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        # Plot modulated signal
        fig.add_trace(
            go.Scatter(x=t[:1000], y=modulated_signal[:1000], name='Modulated', 
                       line=dict(color='green', width=2)),
            row=3, col=1
        )
        
        # Plot demodulated signal
        fig.add_trace(
            go.Scatter(x=t, y=demod_signal, name='Demodulated', 
                       line=dict(color='purple', width=2)),
            row=4, col=1
        )
        
        # Overlay original message
        fig.add_trace(
            go.Scatter(x=t, y=message_signal, name='Original Message', 
                       line=dict(color='blue', width=2, dash='dash'),
                       opacity=0.5),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,  # Slightly increased
            margin=dict(l=50, r=50, t=80, b=50),  # More margin
            title_text=f"{modulation_type} - Signal Analysis",
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Amplitude")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Frequency domain analysis
    st.markdown("---")
    st.markdown("### Frequency Domain Analysis")
    
    freq_col1, freq_col2 = st.columns(2)
    
    with freq_col1:
        # FFT of signals
        fft_message = np.fft.fft(message_signal)
        freqs_message = np.fft.fftfreq(len(message_signal), 1/sample_rate)
        message_data = message_signal
        
        fft_modulated = np.fft.fft(modulated_signal)
        freqs_modulated = np.fft.fftfreq(len(modulated_signal), 1/sample_rate)
        
        # Plot frequency spectra
        fig_freq = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Message Signal Spectrum', 'Modulated Signal Spectrum'),
            vertical_spacing=0.18  # increased for freq domain as well
        )
        
        # Message spectrum
        fig_freq.add_trace(
            go.Scatter(x=freqs_message[:len(freqs_message)//2], 
                       y=np.abs(fft_message[:len(fft_message)//2]),
                       name='Message Spectrum', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Modulated spectrum
        fig_freq.add_trace(
            go.Scatter(x=freqs_modulated[:len(freqs_modulated)//2], 
                       y=np.abs(fft_modulated[:len(fft_modulated)//2]),
                       name='Modulated Spectrum', line=dict(color='green')),
            row=2, col=1
        )
        
        fig_freq.update_layout(
            height=550,
            margin=dict(l=50, r=50, t=50, b=50),
            title_text="Frequency Spectra Comparison",
            template="plotly_white"
        )
        fig_freq.update_xaxes(title_text="Frequency (Hz)")
        fig_freq.update_yaxes(title_text="Magnitude")
        
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with freq_col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Modulation Analysis")
        
        # Calculate bandwidth
        message_bandwidth = 2 * message_freq
        
        if modulation_type == "Amplitude Modulation (AM)":
            modulated_bandwidth = 2 * message_bandwidth
            efficiency = (mod_index**2 / (2 + mod_index**2)) * 100
        elif modulation_type == "Frequency Modulation (FM)":
            modulated_bandwidth = 2 * (mod_index + 1) * message_bandwidth
            efficiency = 100
        elif modulation_type == "Phase Modulation (PM)":
            modulated_bandwidth = 2 * (mod_index + 1) * message_bandwidth
            efficiency = 100
        else:
            modulated_bandwidth = message_bandwidth
            efficiency = 95
        
        st.markdown(f"""
        **Signal Parameters:**
        - Message Frequency: {message_freq:.1f} Hz
        - Carrier Frequency: {carrier_freq:.1f} Hz
        - Modulation Index: {mod_index:.2f}
        - Message Bandwidth: {message_bandwidth:.1f} Hz
        - Modulated Bandwidth: {modulated_bandwidth:.1f} Hz
        - Bandwidth Efficiency: {efficiency:.1f}%
        
        **Signal Powers:**
        - Message Power: {np.mean(message_data**2):.4f}
        - Carrier Power: {np.mean(carrier_signal**2):.4f}
        - Modulated Power: {np.mean(modulated_signal**2):.4f}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Theory section
    st.markdown("---")
    st.markdown("### Modulation Theory")
    
    theory_col1, theory_col2 = st.columns(2)
    
    with theory_col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Amplitude Modulation (AM)")
        st.markdown("""
        - **Equation**: s(t) = [1 + m(t)] * cos(2π f_c t)
        - **Bandwidth**: 2 * Message Bandwidth
        - **Advantages**: Simple demodulation
        - **Disadvantages**: Power inefficient
        - **Applications**: AM radio, aviation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Frequency Modulation (FM)")
        st.markdown("""
        - **Equation**: s(t) = cos(2π f_c t + β sin(2π f_m t))
        - **Bandwidth**: 2 * (β + 1) * f_m (Carson's Rule)
        - **Advantages**: Noise resistant, high fidelity
        - **Disadvantages**: Large bandwidth requirement
        - **Applications**: FM radio, satellite communication
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with theory_col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Phase Modulation (PM)")
        st.markdown("""
        - **Equation**: s(t) = cos(2π f_c t + k_p m(t))
        - **Bandwidth**: Similar to FM, depends on modulation index
        - **Advantages**: Less susceptible to noise than AM
        - **Disadvantages**: More complex demodulation
        - **Applications**: Digital modulation schemes, satellites
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Single Sideband (SSB)")
        st.markdown("""
        - **Equation**: s(t) = m(t) * cos(2π f_c t) ± m_h(t) * sin(2π f_c t)
        - **Bandwidth**: Equal to message bandwidth
        - **Advantages**: Bandwidth efficient
        - **Disadvantages**: Complex demodulation
        - **Applications**: Amateur radio, military communication
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive features
    st.markdown("---")
    st.markdown("### Advanced Analysis")
    
    advanced_col1, advanced_col2 = st.columns(2)
    
    with advanced_col1:
        if st.checkbox("Show Modulation Envelope"):
            if modulation_type == "Amplitude Modulation (AM)":
                envelope = np.abs(scipy.signal.hilbert(modulated_signal))
                
                fig_env = go.Figure()
                fig_env.add_trace(
                    go.Scatter(x=t[:2000], y=modulated_signal[:2000], 
                               name='AM Signal', line=dict(color='blue'))
                )
                fig_env.add_trace(
                    go.Scatter(x=t[:2000], y=envelope[:2000], 
                               name='Envelope', line=dict(color='red', width=3))
                )
                fig_env.add_trace(
                    go.Scatter(x=t[:2000], y=-envelope[:2000], 
                               name='Negative Envelope', line=dict(color='red', width=3))
                )
                
                fig_env.update_layout(
                    title="AM Signal with Envelope",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_env, use_container_width=True)
    
    with advanced_col2:
        if st.checkbox("Show Signal Quality Metrics"):
            signal_power = np.mean(modulated_signal**2)
            noise_estimate = np.std(modulated_signal - np.mean(modulated_signal))
            snr_db = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
            
            fft_mod = np.abs(np.fft.fft(modulated_signal))
            fundamental_power = np.max(fft_mod)**2
            total_power = np.sum(fft_mod**2)
            thd = np.sqrt(max(0, (total_power - fundamental_power) / (fundamental_power + 1e-10))) * 100
            
            st.markdown(f"""
            **Signal Quality Metrics:**
            - **SNR**: {snr_db:.2f} dB
            - **THD**: {thd:.2f}%
            - **Peak Factor**: {np.max(np.abs(modulated_signal)) / np.sqrt(np.mean(modulated_signal**2) + 1e-10):.2f}
            - **RMS Value**: {np.sqrt(np.mean(modulated_signal**2)):.4f}
            """)
    
    # Comparison section
    if st.checkbox("Compare Modulation Types"):
        st.markdown("### Modulation Comparison")
        
        am_signal = (1 + 0.5 * message_signal) * carrier_signal
        fm_phase = 2 * np.pi * np.cumsum(carrier_freq + 2 * message_freq * message_signal) / sample_rate
        fm_signal = carrier_amp * np.sin(fm_phase)
        pm_signal = carrier_amp * np.sin(2 * np.pi * carrier_freq * t + 2 * message_signal)
        
        fig_comp = go.Figure()
        
        fig_comp.add_trace(
            go.Scatter(x=t[:1000], y=am_signal[:1000], name='AM', 
                       line=dict(color='blue'))
        )
        fig_comp.add_trace(
            go.Scatter(x=t[:1000], y=fm_signal[:1000], name='FM', 
                       line=dict(color='red'))
        )
        fig_comp.add_trace(
            go.Scatter(x=t[:1000], y=pm_signal[:1000], name='PM', 
                       line=dict(color='green'))
        )
        
        fig_comp.update_layout(
            title="Modulation Types Comparison",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)

if __name__ == "__main__":
    show()
