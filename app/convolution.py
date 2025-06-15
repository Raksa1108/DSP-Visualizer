import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.signal
from scipy.signal import windows
from scipy.io import wavfile
import io
import soundfile as sf

def show():
    st.markdown('<h2 class="sub-header">🔄 Convolution Operations</h2>', unsafe_allow_html=True)
    
    # Create two columns for controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Signal Parameters")
        
        # Signal source selection
        signal_source = st.selectbox(
            "Select Signal Source",
            ["Generate Signals", "Upload Audio Files"]
        )
        
        if signal_source == "Generate Signals":
            # Signal type selection
            signal_type = st.selectbox(
                "Select Signal Type",
                ["Custom Signals", "Rectangular Pulse", "Exponential", "Sine Wave"]
            )
            
            # Time parameters
            duration = st.slider("Duration (seconds)", 0.1, 5.0, 2.0, 0.1)
            sample_rate = st.slider("Sample Rate (Hz)", 100, 2000, 500, 50)
            
            # Create time vector
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Window function selection
            st.markdown("#### Window Function")
            window_type = st.selectbox(
                "Apply Window Function",
                ["None", "Square", "Sine", "Hamming", "Hanning", "Blackman", "Kaiser"]
            )
            
            # Generate signals based on type
            if signal_type == "Custom Signals":
                st.markdown("#### First Signal")
                freq1 = st.slider("Frequency 1 (Hz)", 0.1, 50.0, 5.0, 0.1)
                amp1 = st.slider("Amplitude 1", 0.1, 2.0, 1.0, 0.1)
                
                st.markdown("#### Second Signal")
                freq2 = st.slider("Frequency 2 (Hz)", 0.1, 50.0, 2.0, 0.1)
                amp2 = st.slider("Amplitude 2", 0.1, 2.0, 1.0, 0.1)
                
                # Generate signals
                x1 = amp1 * np.sin(2 * np.pi * freq1 * t)
                x2 = amp2 * np.sin(2 * np.pi * freq2 * t)
                
            elif signal_type == "Rectangular Pulse":
                pulse_width1 = st.slider("Pulse Width 1", 0.1, 1.0, 0.5, 0.05)
                pulse_width2 = st.slider("Pulse Width 2", 0.1, 1.0, 0.3, 0.05)
                
                # Generate rectangular pulses
                x1 = np.where(t < pulse_width1, 1.0, 0.0)
                x2 = np.where(t < pulse_width2, 1.0, 0.0)
                
            elif signal_type == "Exponential":
                decay1 = st.slider("Decay Rate 1", 0.1, 5.0, 1.0, 0.1)
                decay2 = st.slider("Decay Rate 2", 0.1, 5.0, 2.0, 0.1)
                
                # Generate exponential signals
                x1 = np.exp(-decay1 * t)
                x2 = np.exp(-decay2 * t)
                
            else:  # Sine Wave
                freq1 = st.slider("Frequency 1 (Hz)", 0.1, 20.0, 3.0, 0.1)
                freq2 = st.slider("Frequency 2 (Hz)", 0.1, 20.0, 7.0, 0.1)
                
                # Generate sine waves with windowing
                window1 = np.where(t < duration/3, 1.0, 0.0)
                window2 = np.where(t < duration/2, 1.0, 0.0)
                
                x1 = np.sin(2 * np.pi * freq1 * t) * window1
                x2 = np.sin(2 * np.pi * freq2 * t) * window2
            
            # Apply window function
            if window_type != "None":
                if window_type == "Square":
                    window1 = np.where(t < duration/2, 1.0, 0.0)
                    window2 = window1
                elif window_type == "Sine":
                    window1 = np.sin(np.pi * t / duration)
                    window2 = window1
                elif window_type == "Hamming":
                    window1 = windows.hamming(len(t))
                    window2 = window1
                elif window_type == "Hanning":
                    window1 = windows.hann(len(t))
                    window2 = window1
                elif window_type == "Blackman":
                    window1 = windows.blackman(len(t))
                    window2 = window1
                elif window_type == "Kaiser":
                    beta = st.slider("Kaiser Beta", 0.1, 10.0, 5.0)
                    window1 = windows.kaiser(len(t), beta)
                    window2 = window1
                
                x1 = x1 * window1
                x2 = x2 * window2
            
            # Option to generate and download audio files
            st.markdown("#### Generate Audio Files")
            if st.checkbox("Generate Audio for Signals"):
                col_audio1, col_audio2 = st.columns(2)
                
                with col_audio1:
                    if st.button("Generate Signal 1 Audio"):
                        # Normalize signal
                        x1_normalized = x1 / np.max(np.abs(x1))
                        # Create a buffer for WAV file
                        buffer = io.BytesIO()
                        sf.write(buffer, x1_normalized, sample_rate, format='WAV')
                        buffer.seek(0)
                        st.download_button(
                            label="Download Signal 1 WAV",
                            data=buffer,
                            file_name="signal1.wav",
                            mime="audio/wav"
                        )
                
                with col_audio2:
                    if st.button("Generate Signal 2 Audio"):
                        # Normalize signal
                        x2_normalized = x2 / np.max(np.abs(x2))
                        # Create a buffer for WAV file
                        buffer = io.BytesIO()
                        sf.write(buffer, x2_normalized, sample_rate, format='WAV')
                        buffer.seek(0)
                        st.download_button(
                            label="Download Signal 2 WAV",
                            data=buffer,
                            file_name="signal2.wav",
                            mime="audio/wav"
                        )
        
        else:  # Upload Audio Files
            st.markdown("#### Upload Signals")
            file1 = st.file_uploader("Upload First Signal (WAV)", type=['wav'], key="file1")
            file2 = st.file_uploader("Upload Second Signal (WAV)", type=['wav'], key="file2")
            
            if file1 is not None and file2 is not None:
                try:
                    # Read first audio file
                    sample_rate1, x1 = wavfile.read(file1)
                    if len(x1.shape) > 1:
                        x1 = np.mean(x1, axis=1)  # Convert to mono
                    x1 = x1.astype(float) / np.max(np.abs(x1))  # Normalize
                    
                    # Read second audio file
                    sample_rate2, x2 = wavfile.read(file2)
                    if len(x2.shape) > 1:
                        x2 = np.mean(x2, axis=1)  # Convert to mono
                    x2 = x2.astype(float) / np.max(np.abs(x2))  # Normalize
                    
                    # Ensure same sample rate
                    if sample_rate1 != sample_rate2:
                        st.error("Sample rates of the two audio files must match!")
                        return
                    
                    sample_rate = sample_rate1
                    duration = len(x1) / sample_rate
                    t = np.linspace(0, duration, len(x1))
                    
                    # Truncate to same length if necessary
                    min_length = min(len(x1), len(x2))
                    x1 = x1[:min_length]
                    x2 = x2[:min_length]
                    t = t[:min_length]
                    
                except Exception as e:
                    st.error(f"Error reading audio files: {e}")
                    return
            else:
                st.info("Please upload both audio files to proceed.")
                return
        
        # Convolution type
        conv_mode = st.selectbox(
            "Convolution Mode",
            ["full", "same", "valid"]
        )
        
        # Convolution method
        conv_method = st.selectbox(
            "Convolution Method",
            ["Direct (scipy.signal.convolve)", "FFT (scipy.signal.fftconvolve)"]
        )
    
    with col2:
        # Perform convolution
        if conv_method == "Direct (scipy.signal.convolve)":
            conv_result = scipy.signal.convolve(x1, x2, mode=conv_mode)
        else:
            conv_result = scipy.signal.fftconvolve(x1, x2, mode=conv_mode)
        
        # Create time vector for convolution result
        if conv_mode == "full":
            t_conv = np.linspace(0, 2 * duration, len(conv_result))
        elif conv_mode == "same":
            t_conv = t
        else:  # valid
            conv_length = max(len(x1) - len(x2) + 1, len(x2) - len(x1) + 1)
            conv_length = max(conv_length, 1)  # Ensure at least 1 sample
            t_conv = np.linspace(0, duration, len(conv_result))
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Signal 1: x₁(t)', 'Signal 2: x₂(t)', 
                          'Convolution: x₁(t) * x₂(t)', 'Convolution Properties'),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Plot first signal
        fig.add_trace(
            go.Scatter(x=t, y=x1, name='x₁(t)', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Plot second signal
        fig.add_trace(
            go.Scatter(x=t, y=x2, name='x₂(t)', line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        # Plot convolution result
        fig.add_trace(
            go.Scatter(x=t_conv, y=conv_result, name='x₁(t) * x₂(t)', 
                      line=dict(color='green', width=2)),
            row=3, col=1
        )
        
        # Plot convolution properties visualization
        # Cross-correlation for comparison
        cross_corr = scipy.signal.correlate(x1, x2, mode=conv_mode)
        fig.add_trace(
            go.Scatter(x=t_conv, y=cross_corr, name='Cross-correlation', 
                      line=dict(color='orange', width=2, dash='dash'),
                      opacity=0.7),
            row=4, col=1
        )
        
        # Auto-correlation of first signal
        auto_corr = scipy.signal.correlate(x1, x1, mode='same')
        t_auto = np.linspace(-duration/2, duration/2, len(auto_corr))
        fig.add_trace(
            go.Scatter(x=t_auto, y=auto_corr, name='Auto-correlation x₁', 
                      line=dict(color='purple', width=2, dash='dot'),
                      opacity=0.7),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Convolution Analysis",
            showlegend=True,
            template="plotly_white"
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time (s)", row=4, col=1)
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=4, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Information section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🔍 Convolution Properties")
        st.markdown(f"""
        - **Signal 1 Length**: {len(x1)} samples
        - **Signal 2 Length**: {len(x2)} samples
        - **Convolution Length**: {len(conv_result)} samples
        - **Mode**: {conv_mode}
        - **Method**: {conv_method}
        - **Max Convolution Value**: {np.max(conv_result):.4f}
        - **Energy**: {np.sum(conv_result**2):.4f}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📚 Convolution Theory")
        st.markdown("""
        **Convolution** is a mathematical operation that combines two functions to produce a third function:
        
        - **Continuous**: (f * g)(t) = ∫ f(τ)g(t-τ)dτ
        - **Discrete**: (f * g)[n] = Σ f[m]g[n-m]
        
        **Properties**:
        - Commutative: f * g = g * f
        - Associative: (f * g) * h = f * (g * h)
        - Distributive: f * (g + h) = f * g + f * h
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive demonstration
    st.markdown("---")
    st.markdown("### 🎯 Interactive Convolution Demo")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.markdown("#### Step-by-step Visualization")
        step_demo = st.checkbox("Enable Step-by-step Demo")
        
        if step_demo:
            if len(conv_result) <= 1:
                st.warning("Convolution result is too short for step-by-step visualization. Try a different mode (e.g., 'full' or 'same').")
            else:
                step = st.slider("Convolution Step", 0, len(conv_result)-1, 0)
                
                # Create a visualization showing the convolution at a specific step
                fig_demo = go.Figure()
                
                # Show the first signal (stationary)
                fig_demo.add_trace(
                    go.Scatter(x=t, y=x1, name='x₁(t)', line=dict(color='blue'))
                )
                
                # Compute the correct shift for x2
                dt = t[1] - t[0]  # Time step
                # In convolution, x2 is flipped and shifted. The step corresponds to the convolution output index.
                # Shift ranges from -len(x2) to len(x1)-1 in "full" mode.
                # Adjust shift to start with x2 fully to the left of x1.
                shift_samples = step - (len(x2) - 1)
                shift_time = shift_samples * dt
                t_shifted = t + shift_time
                
                # Flip x2
                x2_flipped = np.flip(x2)
                
                # Plot the flipped and shifted x2
                fig_demo.add_trace(
                    go.Scatter(x=t_shifted, y=x2_flipped, name='x₂(t-τ)', 
                              line=dict(color='red', dash='dash'))
                )
                
                # Highlight the overlapping region
                overlap_mask = (t_shifted >= t[0]) & (t_shifted <= t[-1])
                if np.any(overlap_mask):
                    t_overlap = t_shifted[overlap_mask]
                    x2_overlap = x2_flipped[overlap_mask]
                    x1_overlap = np.interp(t_overlap, t, x1)
                    overlap_product = x1_overlap * x2_overlap
                    fig_demo.add_trace(
                        go.Scatter(x=t_overlap, y=overlap_product, name='Overlap Product',
                                  line=dict(color='green', width=1), opacity=0.3,
                                  fill='tozeroy')
                    )
                
                # Set x-axis range to show the full sliding process
                x_range_min = min(t[0], t_shifted[0])
                x_range_max = max(t[-1], t_shifted[-1])
                fig_demo.update_layout(
                    title=f"Convolution Step {step}",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    template="plotly_white",
                    xaxis_range=[x_range_min, x_range_max]
                )
                
                st.plotly_chart(fig_demo, use_container_width=True)
    
    with demo_col2:
        st.markdown("#### Convolution Applications")
        st.markdown("""
        - **Signal Processing**: Filtering, echo, reverb
        - **Image Processing**: Blurring, edge detection
        - **Machine Learning**: Convolutional neural networks
        - **Probability**: Sum of random variables
        - **System Analysis**: Impulse response
        """)
        
        # Show numerical results
        if st.checkbox("Show Numerical Values"):
            st.markdown("**First 10 Convolution Values:**")
            for i in range(min(10, len(conv_result))):
                st.text(f"y[{i}] = {conv_result[i]:.6f}")
