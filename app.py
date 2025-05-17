import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import pickle
import io

# function for hyperbolic decline

from streamlit_plotly_events import plotly_events

def safe_load_pickle(file):
    """Safely load pickle file with version compatibility handling."""
    try:
        # First try: direct pandas read
        return pd.read_pickle(file)
    except Exception as e1:
        try:
            # Second try: using pickle with latin1 encoding
            return pickle.load(file, encoding='latin1')
        except Exception as e2:
            try:
                # Third try: read as bytes and convert to DataFrame
                raw_data = file.read()
                file_like = io.BytesIO(raw_data)
                return pickle.load(file_like, encoding='latin1')
            except Exception as e3:
                st.error(f"Failed to load pickle file: {str(e3)}")
                return None

@st.cache_resource
def load_model_fn():
    # Get the absolute path to the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best_cnn_model.keras')
    
    try:
        # Try loading with custom_objects and compile=False
        model = load_model(model_path, 
                         custom_objects=None, 
                         compile=False)
        return model
    except Exception as e:
        st.warning(f"Model file not found or couldn't be loaded: {str(e)}")
        st.warning("Peak detection with AI will be disabled.")
        return None

def first_true_in_windows(nums, window=5):
    filtered_nums = []
    last_num = None
    for num in nums:
        if last_num is None or num - last_num >= window:
            filtered_nums.append(num)
            last_num = num

    return filtered_nums

def split_into_windows(arr, window_size=20):
    # Calculate how many complete windows we can make
    num_windows = len(arr) // window_size
    # Take only the data that fits into complete windows
    arr = arr[:num_windows * window_size]
    # Reshape into windows
    chunks = arr.reshape(-1, window_size)
    # Add channel dimension (for the 1 in (None, 20, 1))
    chunks = chunks.reshape(chunks.shape[0], chunks.shape[1], 1)
    # Normalize each window
    normalized_chunks = np.array([(chunk - np.mean(chunk)) / np.std(chunk) for chunk in chunks])
    return normalized_chunks

# Predict peaks using the AI model
def predict_peaks_with_model(data, model):
    values = split_into_windows(data)
    results = []
    for i in values:
        # Reshape to (1, 20, 1) for single prediction
        prediction = model.predict(i.reshape(1, 20, 1))
        results.append(prediction > 0.5)

    return first_true_in_windows(np.where(np.array(results).flatten())[0])


def hyperbolic(t, qi, di, b):
    # Add safety checks to prevent overflow
    t = np.clip(t, 0, 1e6)  # Clip large values
    b = np.clip(b, 0.1, 10)  # Keep b in reasonable range
    return qi * (1 + b * di * t)**(-1/b)

def exponential(t, qi, di):
    return qi * np.exp(-di * t)

def harmonic(t, qi, di):
    return qi / (1 + di * t)


def separate_data_by_peaks(data, peaks):
    segments = []
    for peak in peaks:
        start_idx = max(data.index[0], data.index[peak - 1])
        end_idx = min(data.index[-1], data.index[peak + 1])
        segment = data.loc[start_idx:end_idx]
        segments.append(segment)
    return segments

def fit_curve(x, y, method='hyperbolic'):
    # Normalize data to prevent overflow
    x_normalized = x / max(x)
    y_normalized = y / max(y)
    
    if method == 'hyperbolic':
        # Add bounds to prevent overflow
        bounds = ([0, 0, 0.1], [np.inf, np.inf, 1.0])  # qi, di, b bounds
        popt, _ = curve_fit(hyperbolic, x_normalized, y_normalized, 
                          bounds=bounds, maxfev=100000)
        qi, di, b = popt
        qi = qi * max(y)
        di = di / max(x)
        return lambda t: hyperbolic(t, qi, di, b)
    
    elif method == 'exponential':
        # Add bounds to prevent overflow
        bounds = ([0, 0], [np.inf, np.inf])  # qi, di bounds
        popt, _ = curve_fit(exponential, x_normalized, y_normalized, 
                          bounds=bounds, maxfev=100000)
        qi, di = popt
        qi = qi * max(y)
        di = di / max(x)
        return lambda t: exponential(t, qi, di)
    
    elif method == 'harmonic':
        # Add bounds to prevent overflow
        bounds = ([0, 0], [np.inf, np.inf])  # qi, di bounds
        popt, _ = curve_fit(harmonic, x_normalized, y_normalized, 
                          bounds=bounds, maxfev=100000)
        qi, di = popt
        qi = qi * max(y)
        di = di / max(x)
        return lambda t: harmonic(t, qi, di)

# Function to remove a peak at a given index
def remove_peak(data, index):
    st.session_state.removed_peak_indices.append(index)
    return data[~np.isin(data, [index])]

def reset_peaks():
    st.session_state.peaks = []

# Streamlit app
def main():

    model = load_model_fn()

    if 'removed_peak_indices' not in st.session_state:
        st.session_state['removed_peak_indices'] = []


    st.title("Peak Detection and Decline Curve")

    # Upload dataset
    st.subheader("Upload Pickle File")
    uploaded_file = st.file_uploader("Choose a PKL file", type="pkl")

    if uploaded_file is not None:
        # Read dataset with safe loading
        df = safe_load_pickle(uploaded_file)
        
        if df is None:
            st.error("Failed to load the file. Please check the file format and try again.")
            return
            
        # Reset peaks when new file is uploaded

        if model is not None and st.button('Predict Peaks with AI Model'):
            st.session_state.peaks = predict_peaks_with_model(df['BOE'].values, model)
            st.write("Predicted Peaks (AI Model):", st.session_state.peaks)
        elif model is None:
            st.info("AI model is not available. Please use the Percentile Method for peak detection.")


        # Plot data
        st.subheader("Data Visualization")
        
        # Initialize peaks in session state if not exists
        if 'peaks' not in st.session_state:
            st.session_state.peaks = []

        # Create the figure with interactive points
        fig = go.Figure()

        dfValues = []
        for i in range(len(df)):
            dfValues.append(df['BOE'].values[i])
        
        # Add the main data trace with markers
        fig.add_trace(go.Scatter(
            x=df.index,
            y=dfValues,
            mode='lines',
            name='Data',
        ))
        fitting_method = st.selectbox(
            'Select Fitting Method',
            ['hyperbolic', 'exponential', 'harmonic'],
            help='Choose the decline curve type to fit the data'
        )

        # Add fitted curves if there are peaks
        if st.session_state.peaks:
            segments = []
            start_idx = 0
            for idx in st.session_state.peaks:
                segments.append(df.iloc[start_idx:idx + 1])
                start_idx = idx + 1
            segments.append(df.iloc[start_idx:])
            
            # Only process segments after the first peak
            for i, selected_point in enumerate(segments):
                if i == 0:  # Skip the first segment (before first peak)
                    continue
                    
                t_fit = np.array([t.timestamp() for t in selected_point["BOE"].index])
                q_fit = np.array(selected_point["BOE"].values)
                
                # Create mask for non-zero values
                mask = q_fit > 0
                t_fit_nonzero = t_fit[mask]
                q_fit_nonzero = q_fit[mask]
                
                # Normalize time and production values (only for non-zero values)
                t_fit_norm = (t_fit_nonzero - t_fit_nonzero.min()) / (t_fit_nonzero.max() - t_fit_nonzero.min())
                q_fit_norm = (q_fit_nonzero - q_fit_nonzero.min()) / (q_fit_nonzero.max() - q_fit_nonzero.min())

                # Fit curve using selected method (only on non-zero values)
                fit_func = fit_curve(t_fit_norm, q_fit_norm, method=fitting_method)
                
                # Prepare forecast data for all points (including zeros)
                t_forecast = np.array([t.timestamp() for t in selected_point["BOE"].index])
                t_forecast_norm = (t_forecast - t_fit_nonzero.min()) / (t_fit_nonzero.max() - t_fit_nonzero.min())
                
                # Get predictions and denormalize
                q_forecast_norm = fit_func(t_forecast_norm)
                q_forecast = q_forecast_norm * (q_fit_nonzero.max() - q_fit_nonzero.min()) + q_fit_nonzero.min()

                forecast = []
                for i in range(len(q_forecast)):
                    forecast.append(max(0, q_forecast[i]))
                # Plot curve on the same figure
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(t_forecast, unit='s'),
                    y=forecast,
                    mode='lines',
                    name=f'Fitted Curve ({fitting_method})'
                ))

        fig.update_layout(
            title='Oil and Gas Production Forecast',
            xaxis_title='Date',
            yaxis_title='BOE',
            template='plotly_white',
            hovermode='closest'
        )

        # Add click events to the plot and display it
        selected_points = plotly_events(fig, click_event=True, select_event=True)
            
        # Handle click events
        if selected_points:
            for point in selected_points:
                # Find the closest data point to the clicked point
                clicked_x = pd.to_datetime(point['x'])
                closest_idx = (df.index.get_indexer([clicked_x], method='nearest'))[0]
                
                # Toggle peak: if it exists, remove it; if it doesn't, add it
                if closest_idx in st.session_state.peaks:
                    st.session_state.peaks = [p for p in st.session_state.peaks if p != closest_idx]
                    st.write(f"Removed peak at {df.index[closest_idx]}")
                else:
                    st.session_state.peaks.append(closest_idx)
                    st.write(f"Added peak at {df.index[closest_idx]}")
                st.session_state.peaks.sort()  # Keep peaks in order
                st.experimental_rerun()  # Force rerun to update the graph

        # Display current peaks
        st.write("Current Peaks:")
        for peak in st.session_state.peaks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"- {df.index[peak]}")
            with col2:
                if st.button(f"Remove Peak", key=f"remove_{peak}"):
                    st.session_state.peaks = [p for p in st.session_state.peaks if p != peak]
                    st.experimental_rerun()

        

if __name__ == "__main__":
    main()
