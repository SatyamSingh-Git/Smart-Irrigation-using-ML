import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time # Optional for simulating delay

# --- Configuration ---
st.set_page_config(page_title="Plant Watering Predictor", page_icon="🪴", layout="wide")

# This path should point to the model file saved by the training script
MODEL_PATH = "best_plant_model.joblib"

# Define expected options based on the training data's categories
# Make sure these lists exactly match the unique values in your training data's
# 'Plant_Type' and 'Pot_Size' columns if they differ from the generator script.
PLANT_TYPES = ['Succulent', 'Fern', 'Tropical', 'Herb', 'Flowering']
POT_SIZES = ['Small', 'Medium', 'Large']

# --- Load Model (Cached) ---
# Use st.cache_resource for objects like models that don't change often
# Use st.cache_data for dataframes or serializable objects
# Adjust based on your streamlit version if needed
@st.cache_resource
def load_model(path):
    """Loads the pre-trained pipeline from disk."""
    try:
        pipeline = joblib.load(path)
        print(f"Model loaded successfully from {path}") # Log for debugging
        return pipeline
    except FileNotFoundError:
        st.error(f"Fatal Error: Model file not found at '{path}'.")
        st.error("Please ensure the training script has been run and the model file exists in the correct location.")
        return None
    except ModuleNotFoundError as e:
         st.error(f"Fatal Error: A required library might be missing: {e}")
         st.error("Please ensure scikit-learn, xgboost, lightgbm etc. are installed in the Streamlit environment.")
         return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading the model: {e}")
        return None

print("Attempting to load model...") # Log for debugging
pipeline = load_model(MODEL_PATH)
print("Model loading attempt finished.") # Log for debugging

st.title("🪴 Indoor Plant Watering Needs Predictor")
st.markdown("""
Welcome! This app uses a Machine Learning model to predict if your indoor plant needs watering
based on its type, pot size, and current sensor readings.
""")

# Stop execution if the model failed to load
if pipeline is None:
    st.warning("Application cannot proceed without a valid model file.")
    st.stop()


# --- User Input Section ---
st.sidebar.header("Enter Plant & Sensor Data")

# Use the sidebar for inputs to keep the main area clean for results
plant_type = st.sidebar.selectbox(
    "Select Plant Type:",
    options=PLANT_TYPES,
    index=PLANT_TYPES.index('Tropical') # Sensible default
)

pot_size = st.sidebar.selectbox(
    "Select Pot Size:",
    options=POT_SIZES,
    index=POT_SIZES.index('Medium') # Sensible default
)

st.sidebar.markdown("---") # Separator

soil_moisture = st.sidebar.slider(
    "Soil Moisture (%):",
    min_value=0, max_value=100, value=50, step=1,
    help="Current moisture reading from your soil sensor."
)

temperature = st.sidebar.slider(
    "Ambient Temperature (°C):",
    min_value=10.0, max_value=35.0, value=22.0, step=0.5, format="%.1f",
    help="Temperature near the plant."
)

humidity = st.sidebar.slider(
    "Ambient Humidity (%):",
    min_value=20, max_value=90, value=60, step=1,
    help="Relative humidity near the plant."
)

light_level = st.sidebar.slider(
    "Light Level (lux):",
    min_value=0, max_value=25000, value=5000, step=100,
    help="Approximate light intensity the plant is receiving."
)


# --- Prediction Logic ---
st.header("Prediction")

# Create a button to trigger prediction
if st.button("Predict Watering Need", type="primary", use_container_width=True):

    # 1. Prepare Input Data
    input_data = pd.DataFrame({
        # Column names MUST match those used during training X
        'Plant_Type': [plant_type],
        'Pot_Size': [pot_size],
        'Soil_Moisture': [soil_moisture],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Light_Level': [light_level]
    })

    st.write("User Input Summary:")
    st.dataframe(input_data, use_container_width=True)

    # 2. Make Prediction
    try:
        with st.spinner('Analyzing sensor data...'):
            start_pred_time = time.time()
            prediction = pipeline.predict(input_data)[0] # Get the single prediction (0 or 1)

            # Get probabilities (useful for confidence)
            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                 probability = pipeline.predict_proba(input_data)[0] # [prob_class_0, prob_class_1]
                 prob_water = probability[1]
                 prob_no_water = probability[0]
            else:
                 prob_water = None # Model doesn't support probabilities
                 prob_no_water = None

            pred_time = time.time() - start_pred_time
            # time.sleep(1) # Optional: simulate processing time

        # 3. Display Result
        st.subheader("Recommendation:")
        col1, col2 = st.columns([1, 3]) # Ratio for icon and text

        with col1:
             if prediction == 1:
                 st.image("https://emojicdn.elk.sh/💧", width=80) # Water droplet emoji
             else:
                 st.image("https://emojicdn.elk.sh/✅", width=80) # Checkmark emoji

        with col2:
            if prediction == 1:
                st.success("**WATER the plant.**")
                if prob_water is not None:
                    st.metric("Confidence", f"{prob_water:.1%}")
                st.caption(f"Prediction made in {pred_time:.3f} seconds.")
            else:
                st.error("**DO NOT water the plant.**")
                if prob_no_water is not None:
                    st.metric("Confidence", f"{prob_no_water:.1%}")
                st.caption(f"Prediction made in {pred_time:.3f} seconds.")

            # Display probabilities if available
            if prob_water is not None:
                 st.write("Detailed Probabilities:")
                 st.json({
                     "Probability Don't Water (0)": f"{prob_no_water:.4f}",
                     "Probability Water (1)": f"{prob_water:.4f}"
                 })

    except Exception as e:
        st.error(f"An error occurred during prediction:")
        st.exception(e) # Display the full error details in the app

else:
     st.info("Click the button above to get a watering recommendation based on the inputs.")


# --- Optional: Add Info/Help in Expander ---
with st.expander("ℹ️ About this App & Model"):
    st.markdown(f"""
        This application uses a **`{type(pipeline.named_steps['classifier']).__name__}`** model,
        loaded from the file `{MODEL_PATH}`.

        The model was trained on synthetic data simulating various indoor plant conditions.
        The pipeline includes preprocessing steps (scaling numerical data and one-hot encoding categorical data)
        before feeding the data to the model.

        **Input Features:**
        *   **Plant Type:** Categorical (e.g., Fern, Succulent)
        *   **Pot Size:** Categorical (Small, Medium, Large)
        *   **Soil Moisture:** Numerical (%) - Lower means drier.
        *   **Temperature:** Numerical (°C) - Ambient temperature.
        *   **Humidity:** Numerical (%) - Ambient relative humidity.
        *   **Light Level:** Numerical (lux) - Light intensity.

        **Output:**
        *   A binary prediction (Water / Don't Water).
        *   A confidence score (probability) if the model supports it.

        **Disclaimer:** This is a demonstration based on a generated dataset. Always use your judgment and visually inspect your plants for the most accurate assessment of watering needs.
    """)