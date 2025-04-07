import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn # Need this for loading sklearn objects with pickle

# --- Configuration ---
st.set_page_config(page_title="Smart Irrigation Predictor", page_icon="💧", layout="centered")

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ML_IOT.pkl")  #

LABEL_ENCODER_PATH = 'label_encoder.pkl'
SCALER_PATH = 'scaler.pkl'

# --- Load Artifacts (Model, Encoder, Scaler) ---
@st.cache_resource # Cache resource loading
def load_artifacts(model_path, encoder_path, scaler_path):
    """Loads the pre-trained model, label encoder, and scaler."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(encoder_path, 'rb') as file:
            label_encoder = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        print("Artifacts loaded successfully.")
        return model, label_encoder, scaler
    except FileNotFoundError as e:
        st.error(f"Fatal Error: Could not find required file: {e}. ")
        st.error("Please ensure the training script ran successfully and the .pkl files are in the same directory.")
        return None, None, None
    except ModuleNotFoundError as e:
         st.error(f"Fatal Error: A required library might be missing: {e}")
         st.error("Please ensure scikit-learn etc. are installed in the Streamlit environment.")
         return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred loading artifacts: {e}")
        return None, None, None

# Load the artifacts
model, label_encoder, scaler = load_artifacts(MODEL_PATH, LABEL_ENCODER_PATH, SCALER_PATH)

# Get the list of known crop types from the loaded encoder
if label_encoder:
    KNOWN_CROP_TYPES = list(label_encoder.classes_)
else:
    KNOWN_CROP_TYPES = ["Error loading types"] # Fallback

df = pd.read_csv("data set for smart irrigation sysytem.csv")
df.columns = ['Temperature', 'Humidity', 'Crop_Type', 'Pump_Time', 'Soil_moisture']


# --- App Layout & Title ---

st.title("💧 Smart Irrigation - Soil Moisture Predictor")
st.markdown("""
This app predicts whether the soil moisture is **Low (0)** or **High (1)**
based on sensor readings and crop type, using a trained Machine Learning model.
*(Based on the provided dataset where Soil Moisture > 50 was classified as High)*
""")

# Stop if artifacts failed to load
if not model or not label_encoder or not scaler:
    st.warning("Application cannot proceed without loading necessary model files.")
    st.stop()

# --- User Input Section ---
st.sidebar.header("Enter Current Conditions")

# Get numerical inputs
temperature = st.sidebar.slider(
    "Temperature (°C):",
    min_value=float(df['Temperature'].min()) - 5, # Use data bounds +/- buffer
    max_value=float(df['Temperature'].max()) + 5,
    value=27.5, # Example default
    step=0.5,
    format="%.1f"
)

humidity = st.sidebar.slider(
    "Humidity (%):",
    min_value=float(df['Humidity'].min()) - 10,
    max_value=100.0,
    value=90.0, # Example default
    step=1.0,
    format="%.1f"
)

pump_time = st.sidebar.number_input(
    "Pump Time (e.g., minutes in last cycle):",
    min_value=0.0,
    max_value=float(df['Pump_Time'].max()) + 10, # Use max from data + buffer
    value=5.0, # Example default
    step=1.0,
    format="%.1f",
    help="Time the pump was running previously. Use 0 if unsure or not relevant to prediction task as designed."
)

# Get categorical input
crop_type = st.sidebar.selectbox(
    "Select Crop Type:",
    options=KNOWN_CROP_TYPES,
    index=KNOWN_CROP_TYPES.index('Ground Nut') if 'Ground Nut' in KNOWN_CROP_TYPES else 0 # Default
)

# --- Prediction Logic ---
st.header("Prediction Result")

# Button to trigger prediction
if st.button("Predict Soil Moisture Level", type="primary"):

    # 1. Prepare Input Data
    # Create a DataFrame first to easily display input
    input_df_display = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Crop_Type': [crop_type],
        'Pump_Time': [pump_time]
    })
    st.write("Input Data:")
    st.dataframe(input_df_display, hide_index=True)

    try:
        # Encode the selected crop type
        crop_encoded = label_encoder.transform([crop_type])[0]

        # Create the input array for the scaler/model in the correct order
        # Order MUST match the order of columns in X during training before scaling
        input_array = np.array([[temperature, humidity, crop_encoded, pump_time]])

        # 2. Scale the input data
        input_scaled = scaler.transform(input_array)

        # 3. Make Prediction
        with st.spinner("🧠 Predicting..."):
            prediction = model.predict(input_scaled)[0] # Get the single prediction
            # Get probabilities if the model supports it (Random Forest does)
            probabilities = model.predict_proba(input_scaled)[0] # [Prob_Class_0, Prob_Class_1]
            confidence = probabilities[prediction] # Confidence in the predicted class

        # 4. Display Result
        st.subheader("Predicted Soil Moisture Status:")

        if prediction == 1:
            st.success(f"**High Moisture (Class 1)**")
            st.progress(confidence)
            st.metric("Confidence", f"{confidence:.1%}")
        else:
            st.error(f"**Low Moisture (Class 0)**")
            st.progress(confidence)
            st.metric("Confidence", f"{confidence:.1%}")

        # Expander for more details
        with st.expander("Show Probabilities"):
             st.write(f"Probability of Low Moisture (0): {probabilities[0]:.4f}")
             st.write(f"Probability of High Moisture (1): {probabilities[1]:.4f}")

    except ValueError as ve:
         if "Found input variables with inconsistent numbers of samples" in str(ve):
              st.error("Error during scaling. Please ensure all inputs are provided correctly.")
              st.error(f"Details: {ve}")
         elif "y contains previously unseen labels" in str(ve):
              st.error(f"Error: The crop type '{crop_type}' was not seen during training.")
              st.error("Please select a crop type from the dropdown list.")
         else:
             st.error(f"An unexpected Value Error occurred during preprocessing or prediction: {ve}")
    except Exception as e:
        st.error("An error occurred during prediction:")
        st.exception(e) # Show detailed traceback

else:
     st.info("Adjust the inputs in the sidebar and click the button to predict.")


# --- Optional: Add Info/Help in Expander ---
with st.expander("ℹ️ About this Application"):
    st.markdown(f"""
        This app uses a **`{type(model).__name__}`** model loaded from `{MODEL_PATH}`.
        It predicts soil moisture level based on the inputs provided.

        **Preprocessing Steps Applied:**
        1.  **Label Encoding:** The selected 'Crop Type' is converted into a numerical value using a LabelEncoder (`{LABEL_ENCODER_PATH}`) fitted on the original training data. The mapping is:
            ```
            {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}
            ```
        2.  **Feature Scaling:** All input features (Temperature, Humidity, Encoded Crop Type, Pump Time) are scaled using a StandardScaler (`{SCALER_PATH}`) fitted on the original training data.

        **Prediction:**
        *   **Class 0:** Indicates Low Soil Moisture (<= 50 based on training script threshold).
        *   **Class 1:** Indicates High Soil Moisture (> 50 based on training script threshold).

        *Model trained on the dataset `data set for smart irrigation sysytem.csv`.*
    """)