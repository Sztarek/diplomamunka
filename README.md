# Decline Curve Analysis Tool

This Streamlit application provides an interactive tool for analyzing oil and gas production data, detecting peaks, and fitting decline curves.

## Main Features

### 1. Data Loading and Processing
- Supports loading production data from pickle files
- Safely handles different pickle file formats and encodings

### 2. AI-Powered Peak Detection
- Uses a pre-trained TensorFlow model to detect production peaks
- Processes data in windows of 20 points for prediction

### 3. Decline Curve Fitting
Supports three types of decline curves:

1. **Hyperbolic Decline**
```python
def hyperbolic(t, qi, di, b):
    return qi * (1 + b * di * t)**(-1/b)
```

2. **Exponential Decline**
```python
def exponential(t, qi, di):
    return qi * np.exp(-di * t)
```

3. **Harmonic Decline**
```python
def harmonic(t, qi, di):
    return qi / (1 + di * t)
```

### 4. Interactive Visualization
- Interactive plot using Plotly
- Click-to-add/remove peaks
- Real-time curve fitting visualization

### 5. Data Normalization
- Z-score normalization for AI model input
- Min-max normalization for curve fitting

## Usage

1. Upload a pickle file containing production data
2. Choose between AI-based peak detection or manual peak selection
3. Select a decline curve fitting method (hyperbolic, exponential, or harmonic)
4. Interact with the plot to add/remove peaks
5. View the fitted decline curves for each segment

## Dependencies
- streamlit
- pandas
- numpy
- plotly
- scipy
- tensorflow
- streamlit-plotly-events

## Note
The application requires a pre-trained model file (`best_cnn_model.keras`) to enable AI-based peak detection. If the model is not available, the application will fall back to manual peak detection methods.
