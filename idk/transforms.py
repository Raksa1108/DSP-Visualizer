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

def show():
    st.title("🔄 Signal Transforms")
    st.markdown("Explore different signal transforms and their properties")
    
    # Create tabs for different transforms
    tab1, tab2, tab3, tab4 = st.tabs(["DTFT", "CTFT", "CTFS", "DTFS"])
    
    with tab1:
        dtft_tab()
    
    with tab2:
        ctft_tab()
    
    with tab3:
        ctfs_tab()
    
    with tab4:
        dtfs_tab()

def dtft_tab():
    st.header("Discrete-Time Fourier Transform (DTFT)")
    st.markdown("The DTFT transforms discrete-time signals into continuous frequency domain")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Signal Input")
        
        # Signal type selection
        signal_type = st.selectbox("Select Signal Type", 
                                 ["Unit Impulse", "Unit Step", "Exponential", "Sinusoidal", "Upload Audio"],
                                 key="dtft_signal_type")
        
        # Parameters based on signal type
        if signal_type == "Unit Impulse":
            n_samples = st.slider("Number of samples", 10, 100, 50, key="dtft_n_samples")
            delay = st.slider("Delay (samples)", 0, n_samples//2, 0, key="dtft_delay")
            n = np.arange(-n_samples//2, n_samples//2)
            x = np.zeros(len(n))
            x[n == delay] = 1
            
        elif signal_type == "Unit Step":
            n_samples = st.slider("Number of samples", 10, 100, 50, key="dtft_n_samples_step")
            n = np.arange(0, n_samples)
            x = np.ones(len(n))
            
        elif signal_type == "Exponential":
            n_samples = st.slider("Number of samples", 10, 100, 50, key="dtft_n_samples_exp")
            alpha = st.slider("Decay factor (α)", 0.1, 0.9, 0.5, key="dtft_alpha")
            n = np.arange(0, n_samples)
            x = np.power(alpha, n)
            
        elif signal_type == "Sinusoidal":
            n_samples = st.slider("Number of samples", 10, 200, 100, key="dtft_n_samples_sin")
            freq = st.slider("Frequency (cycles/sample)", 0.01, 0.5, 0.1, key="dtft_freq")
            phase = st.slider("Phase (radians)", 0.0, 2*np.pi, 0.0, key="dtft_phase")
            n = np.arange(0, n_samples)
            x = np.sin(2 * np.pi * freq * n + phase)
            
        elif signal_type == "Upload Audio":
            uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'], key="dtft_audio")
            if uploaded_file is not None:
                try:
                    # Read audio file
                    fs, x = wavfile.read(uploaded_file)
                    # Take first channel if stereo
                    if len(x.shape) > 1:
                        x = x[:, 0]
                    # Normalize
                    x = x / np.max(np.abs(x))
                    # Take first 1000 samples for display
                    x = x[:1000]
                    n = np.arange(len(x))
                    st.success(f"Audio loaded: {len(x)} samples")
                except:
                    st.error("Error loading audio file")
                    n = np.arange(50)
                    x = np.zeros(50)
            else:
                n = np.arange(50)
                x = np.zeros(50)
        
        # Plot time domain signal
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=n, y=x, mode='markers+lines', name='x[n]'))
        fig_time.update_layout(title="Time Domain Signal", xaxis_title="n", yaxis_title="x[n]")
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("DTFT Analysis")
        
        # Compute DTFT (approximation using zero-padding)
        N_fft = 1024
        X = np.fft.fft(x, N_fft)
        omega = np.linspace(-np.pi, np.pi, N_fft)
        
        # Magnitude and Phase plots
        fig_dtft = make_subplots(rows=2, cols=1, 
                               subplot_titles=['Magnitude Spectrum', 'Phase Spectrum'])
        
        # Magnitude
        fig_dtft.add_trace(go.Scatter(x=omega, y=np.abs(np.fft.fftshift(X)), 
                                    mode='lines', name='|X(e^jω)|'), row=1, col=1)
        
        # Phase
        fig_dtft.add_trace(go.Scatter(x=omega, y=np.angle(np.fft.fftshift(X)), 
                                    mode='lines', name='∠X(e^jω)'), row=2, col=1)
        
        fig_dtft.update_xaxes(title_text="Frequency (ω)", row=2, col=1)
        fig_dtft.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig_dtft.update_yaxes(title_text="Phase (rad)", row=2, col=1)
        fig_dtft.update_layout(height=600, title="DTFT Representation")
        
        st.plotly_chart(fig_dtft, use_container_width=True)

def ctft_tab():
    st.header("Continuous-Time Fourier Transform (CTFT)")
    st.markdown("The CTFT transforms continuous-time signals into frequency domain")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Signal Input")
        
        signal_type = st.selectbox("Select Signal Type", 
                                 ["Rectangular Pulse", "Exponential Decay", "Gaussian", "Triangular"],
                                 key="ctft_signal_type")
        
        # Time vector
        duration = st.slider("Signal Duration (seconds)", 1, 10, 5, key="ctft_duration")
        fs = st.slider("Sampling Rate (Hz)", 100, 1000, 500, key="ctft_fs")
        t = np.linspace(-duration/2, duration/2, int(fs * duration))
        
        if signal_type == "Rectangular Pulse":
            width = st.slider("Pulse Width (seconds)", 0.1, duration/2, 1.0, key="ctft_pulse_width")
            x = np.where(np.abs(t) <= width/2, 1, 0)
            
        elif signal_type == "Exponential Decay":
            alpha = st.slider("Decay rate", 0.1, 5.0, 1.0, key="ctft_alpha")
            x = np.exp(-alpha * np.abs(t))
            
        elif signal_type == "Gaussian":
            sigma = st.slider("Standard deviation", 0.1, 2.0, 0.5, key="ctft_sigma")
            x = np.exp(-t**2 / (2 * sigma**2))
            
        elif signal_type == "Triangular":
            width = st.slider("Base Width (seconds)", 0.1, duration/2, 1.0, key="ctft_tri_width")
            x = np.maximum(0, 1 - 2*np.abs(t)/width)
        
        # Plot time domain signal
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t, y=x, mode='lines', name='x(t)'))
        fig_time.update_layout(title="Time Domain Signal", xaxis_title="Time (s)", yaxis_title="x(t)")
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("CTFT Analysis")
        
        # Compute FFT (approximation of CTFT)
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x), 1/fs)
        
        # Sort frequencies for plotting
        idx = np.argsort(freqs)
        freqs = freqs[idx]
        X = X[idx]
        
        # Magnitude and Phase plots
        fig_ctft = make_subplots(rows=2, cols=1, 
                               subplot_titles=['Magnitude Spectrum', 'Phase Spectrum'])
        
        # Magnitude
        fig_ctft.add_trace(go.Scatter(x=freqs, y=np.abs(X), 
                                    mode='lines', name='|X(jω)|'), row=1, col=1)
        
        # Phase
        fig_ctft.add_trace(go.Scatter(x=freqs, y=np.angle(X), 
                                    mode='lines', name='∠X(jω)'), row=2, col=1)
        
        fig_ctft.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig_ctft.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig_ctft.update_yaxes(title_text="Phase (rad)", row=2, col=1)
        fig_ctft.update_layout(height=600, title="CTFT Representation")
        
        st.plotly_chart(fig_ctft, use_container_width=True)

def ctfs_tab():
    st.header("Continuous-Time Fourier Series (CTFS)")
    st.markdown("The CTFS represents periodic continuous-time signals as sum of harmonics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Signal Input")
        
        signal_type = st.selectbox("Select Periodic Signal", 
                                 ["Square Wave", "Sawtooth Wave", "Triangle Wave", "Custom Harmonics"],
                                 key="ctfs_signal_type")
        
        # Common parameters
        period = st.slider("Period (seconds)", 0.1, 2.0, 1.0, key="ctfs_period")
        fs = st.slider("Sampling Rate (Hz)", 100, 1000, 500, key="ctfs_fs")
        num_periods = st.slider("Number of periods to display", 1, 5, 3, key="ctfs_num_periods")
        
        t = np.linspace(0, num_periods * period, int(fs * num_periods * period))
        
        if signal_type == "Square Wave":
            duty_cycle = st.slider("Duty Cycle (%)", 10, 90, 50, key="ctfs_duty_cycle")
            x = scipy.signal.square(2 * np.pi * t / period, duty=duty_cycle/100)
            
        elif signal_type == "Sawtooth Wave":
            x = scipy.signal.sawtooth(2 * np.pi * t / period)
            
        elif signal_type == "Triangle Wave":
            x = scipy.signal.sawtooth(2 * np.pi * t / period, 0.5)
            
        elif signal_type == "Custom Harmonics":
            st.subheader("Harmonic Components")
            num_harmonics = st.slider("Number of harmonics", 1, 10, 3, key="ctfs_num_harmonics")
            x = np.zeros_like(t)
            
            for k in range(1, num_harmonics + 1):
                amp = st.slider(f"Harmonic {k} Amplitude", 0.0, 2.0, 1.0/k, key=f"ctfs_amp_{k}")
                phase = st.slider(f"Harmonic {k} Phase (rad)", 0.0, 2*np.pi, 0.0, key=f"ctfs_phase_{k}")
                x += amp * np.sin(2 * np.pi * k * t / period + phase)
        
        # Plot time domain signal
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t, y=x, mode='lines', name='x(t)'))
        fig_time.update_layout(title="Periodic Time Domain Signal", 
                             xaxis_title="Time (s)", yaxis_title="x(t)")
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("CTFS Analysis")
        
        # Compute Fourier Series coefficients
        # Take one period for analysis
        one_period_samples = int(fs * period)
        x_period = x[:one_period_samples]
        t_period = t[:one_period_samples]
        
        # Number of harmonics to compute
        N_harmonics = st.slider("Number of harmonics to display", 5, 50, 20, key="ctfs_n_harmonics")
        
        # Compute coefficients using FFT
        X_fft = np.fft.fft(x_period, 2*N_harmonics)
        freqs = np.fft.fftfreq(2*N_harmonics, 1/fs)
        
        # Extract positive frequencies and coefficients
        pos_idx = freqs > 0
        pos_freqs = freqs[pos_idx][:N_harmonics]
        pos_coeffs = X_fft[pos_idx][:N_harmonics]
        
        # DC component
        dc_coeff = X_fft[0] / len(x_period)
        
        # Normalize coefficients
        coeffs_norm = 2 * pos_coeffs / len(x_period)
        
        # Plot spectrum
        fig_ctfs = make_subplots(rows=2, cols=1, 
                               subplot_titles=['Magnitude Spectrum', 'Phase Spectrum'])
        
        # Add DC component
        harmonics = np.arange(0, N_harmonics + 1)
        magnitudes = np.concatenate([[np.abs(dc_coeff)], np.abs(coeffs_norm)])
        phases = np.concatenate([[np.angle(dc_coeff)], np.angle(coeffs_norm)])
        
        # Magnitude
        fig_ctfs.add_trace(go.Bar(x=harmonics, y=magnitudes, name='|Cn|'), row=1, col=1)
        
        # Phase
        fig_ctfs.add_trace(go.Bar(x=harmonics, y=phases, name='∠Cn'), row=2, col=1)
        
        fig_ctfs.update_xaxes(title_text="Harmonic Number (n)", row=2, col=1)
        fig_ctfs.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig_ctfs.update_yaxes(title_text="Phase (rad)", row=2, col=1)
        fig_ctfs.update_layout(height=600, title="CTFS Coefficients")
        
        st.plotly_chart(fig_ctfs, use_container_width=True)
        
        # Reconstruction
        st.subheader("Signal Reconstruction")
        n_recon = st.slider("Number of harmonics for reconstruction", 1, N_harmonics, 5, key="ctfs_n_recon")
        
        # Reconstruct signal
        x_recon = np.real(dc_coeff) * np.ones_like(t)
        for k in range(1, n_recon + 1):
            if k <= len(coeffs_norm):
                x_recon += 2 * np.real(coeffs_norm[k-1] * np.exp(1j * 2 * np.pi * k * t / period))
        
        fig_recon = go.Figure()
        fig_recon.add_trace(go.Scatter(x=t, y=x, mode='lines', name='Original', opacity=0.7))
        fig_recon.add_trace(go.Scatter(x=t, y=x_recon, mode='lines', name='Reconstructed'))
        fig_recon.update_layout(title=f"Signal Reconstruction with {n_recon} harmonics",
                              xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig_recon, use_container_width=True)

def dtfs_tab():
    st.header("Discrete-Time Fourier Series (DTFS)")
    st.markdown("The DTFS represents periodic discrete-time signals as a sum of complex exponentials")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Signal Input")
        
        # Common parameters
        N = st.slider("Period (samples)", 8, 64, 16, key="dtfs_period")
        num_periods = st.slider("Number of periods to display", 1, 5, 3, key="dtfs_num_periods")
        n = np.arange(0, N * num_periods)
        
        signal_type = st.selectbox("Select Periodic Signal", 
                                 ["Square Wave", "Sawtooth Wave", "Triangle Wave", "Custom Harmonics"],
                                 key="dtfs_signal_type")
        
        if signal_type == "Square Wave":
            duty_cycle = st.slider("Duty Cycle (%)", 10, 90, 50, key="dtfs_duty_cycle")
            x_one_period = np.concatenate([np.ones(int(N * duty_cycle/100)), np.zeros(N - int(N * duty_cycle/100))])
            x = np.tile(x_one_period, num_periods)
            
        elif signal_type == "Sawtooth Wave":
            x_one_period = np.linspace(0, 1, N, endpoint=False)
            x = np.tile(x_one_period, num_periods)
            
        elif signal_type == "Triangle Wave":
            x_one_period = np.concatenate([np.linspace(0, 1, N//2 + 1), np.linspace(1, 0, N - N//2 - 1)])
            x = np.tile(x_one_period, num_periods)
            
        elif signal_type == "Custom Harmonics":
            st.subheader("Harmonic Components")
            num_harmonics = st.slider("Number of harmonics", 1, N//2, 3, key="dtfs_num_harmonics")
            x = np.zeros(N * num_periods)
            for k in range(1, num_harmonics + 1):
                amp = st.slider(f"Harmonic {k} Amplitude", 0.0, 2.0, 1.0/k, key=f"dtfs_amp_{k}")
                phase = st.slider(f"Harmonic {k} Phase (rad)", 0.0, 2*np.pi, 0.0, key=f"dtfs_phase_{k}")
                x += amp * np.cos(2 * np.pi * k * n / N + phase)
        
        # Plot time domain signal
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=n, y=x, mode='markers+lines', name='x[n]'))
        fig_time.update_layout(title="Periodic Discrete-Time Signal", 
                             xaxis_title="n", yaxis_title="x[n]")
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("DTFS Analysis")
        
        # Take one period for analysis
        x_period = x[:N]
        
        # Number of harmonics to compute
        N_harmonics = st.slider("Number of harmonics to display", 1, N//2, min(5, N//2), key="dtfs_n_harmonics")
        
        # Compute DTFS coefficients using FFT
        X = np.fft.fft(x_period, N)
        
        # Extract coefficients
        harmonics = np.arange(0, N_harmonics + 1)
        X = X[:N_harmonics + 1] / N  # Normalize by period
        
        # Plot spectrum
        fig_dtfs = make_subplots(rows=2, cols=1, 
                               subplot_titles=['Magnitude Spectrum', 'Phase Spectrum'])
        
        # Magnitude
        fig_dtfs.add_trace(go.Bar(x=harmonics, y=np.abs(X), name='|Ck|'), row=1, col=1)
        
        # Phase
        fig_dtfs.add_trace(go.Bar(x=harmonics, y=np.angle(X), name='∠Ck'), row=2, col=1)
        
        fig_dtfs.update_xaxes(title_text="Harmonic Number (k)", row=2, col=1)
        fig_dtfs.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig_dtfs.update_yaxes(title_text="Phase (rad)", row=2, col=1)
        fig_dtfs.update_layout(height=600, title="DTFS Coefficients")
        
        st.plotly_chart(fig_dtfs, use_container_width=True)
        
        # Reconstruction
        st.subheader("Signal Reconstruction")
        n_recon = st.slider("Number of harmonics for reconstruction", 1, N_harmonics, min(5, N_harmonics), key="dtfs_n_recon")
        
        # Reconstruct signal
        x_recon = np.zeros_like(n, dtype=float)
        for k in range(n_recon + 1):
            x_recon += 2 * np.real(X[k] * np.exp(1j * 2 * np.pi * k * n / N))
        
        fig_recon = go.Figure()
        fig_recon.add_trace(go.Scatter(x=n, y=x, mode='markers+lines', name='Original', opacity=0.7))
        fig_recon.add_trace(go.Scatter(x=n, y=x_recon, mode='markers+lines', name='Reconstructed'))
        fig_recon.update_layout(title=f"Signal Reconstruction with {n_recon} harmonics",
                              xaxis_title="n", yaxis_title="Amplitude")
        st.plotly_chart(fig_recon, use_container_width=True)
