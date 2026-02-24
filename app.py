import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Import the PyTorch module we built
from image_inference import pneumonia_model, predict_pneumonia

st.set_page_config(page_title="Disease Diagnosis System", page_icon="⚕️", layout="wide")

st.title("⚕️ Multi-Disease Diagnosis System")
st.markdown("Use this system to diagnose Diabetes, Breast Cancer, or Pneumonia using patient records or medical imaging.")

# Sidebar navigation
disease_type = st.sidebar.selectbox(
    "Select Disease to Diagnose",
    ["Breast Cancer", "Diabetes", "Pneumonia"]
)

# Paths to models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
BREAST_CANCER_MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_model.pkl")
DIABETES_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")

# Helper function to load tabular models
@st.cache_resource
def load_tabular_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ------------- Breast Cancer Logic -------------
if disease_type == "Breast Cancer":
    st.header("Breast Cancer Diagnosis")
    st.markdown("Enter patient record details extracted from breast mass imaging to predict if the tumor is **Malignant** or **Benign**.")
    
    model_data = load_tabular_model(BREAST_CANCER_MODEL_PATH)
    if model_data is None:
        st.error(f"Cannot find the trained model at `{BREAST_CANCER_MODEL_PATH}`. Please run `train_tabular.py` first.")
    else:
        model = model_data["model"]
        feature_names = model_data["feature_names"]
        target_names = model_data["target_names"] # ['malignant' 'benign']
        
        # We will use the top 5 most important features for simplicity in the UI,
        # but the standard dataset requires all 30.
        # So we'll auto-fill the median values for 25 features and let user tweak the top 5.
        
        importances = model.feature_importances_
        # Sort and get top 5 indices
        top_indices = np.argsort(importances)[::-1][:5]
        top_feature_names = [feature_names[i] for i in top_indices]
        
        st.subheader("Key Measurements")
        col1, col2 = st.columns(2)
        
        user_inputs = {}
        for i, fname in enumerate(top_feature_names):
            with (col1 if i % 2 == 0 else col2):
                # Standard typical ranges for these features just for UI mock sliders
                user_inputs[fname] = st.slider(f"{fname}", 0.0, 50.0, 15.0, 0.1)
                
        if st.button("Diagnose Breast Cancer"):
            # Create a full feature array with zeros (or realistic medians normally)
            # This is simplified for the demo since the model expects 30 inputs.
            full_features = np.zeros((1, len(feature_names)))
            for i, fname in enumerate(feature_names):
                if fname in user_inputs:
                    full_features[0, i] = user_inputs[fname]
                else:
                    # Provide an arbitrary mean/median safe value for unspecified features
                    full_features[0, i] = 1.0 # placeholder
            
            prediction_idx = model.predict(full_features)[0]
            probability = np.max(model.predict_proba(full_features)[0])
            
            result_class = target_names[prediction_idx]
            
            st.divider()
            if result_class == 'malignant':
                st.error(f"🚨 **Diagnosis: {result_class.upper()}** (Confidence: {probability:.2%})")
                st.markdown("Immediate medical consultation is recommended.")
            else:
                st.success(f"✅ **Diagnosis: {result_class.upper()}** (Confidence: {probability:.2%})")
                st.markdown("The mass appears to be benign. Continue regular screenings.")

# ------------- Diabetes Logic -------------
elif disease_type == "Diabetes":
    st.header("Diabetes Diagnosis")
    st.markdown("Enter patient medical records to predict the likelihood of a positive diabetes test.")
    
    model_data = load_tabular_model(DIABETES_MODEL_PATH)
    if model_data is None:
        st.error(f"Cannot find the trained model at `{DIABETES_MODEL_PATH}`. Please run `train_tabular.py` first.")
    else:
        model = model_data["model"]
        feature_names = model_data["feature_names"]
        target_names = model_data["target_names"] # ['Negative', 'Positive']
        
        st.subheader("Patient Vitals")
        col1, col2 = st.columns(2)
        
        # Pima Diabetes exact feature order
        # ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']
        preg = col1.number_input("Pregnancies", 0, 20, 1)
        plas = col2.slider("Glucose (plas)", 0, 200, 100)
        pres = col1.slider("Blood Pressure (pres)", 0, 140, 70)
        skin = col2.slider("Skin Thickness", 0, 100, 20)
        insu = col1.slider("Insulin", 0, 900, 80)
        mass = col2.slider("BMI (mass)", 0.0, 70.0, 25.0, 0.1)
        pedi = col1.slider("Diabetes Pedigree Function (pedi)", 0.0, 2.5, 0.5, 0.01)
        age = col2.number_input("Age", 1, 120, 30)

        if st.button("Diagnose Diabetes"):
            # Ensure proper feature order
            # The exact names depend on dataset fetch, but typically match the load order
            features_dict = {
                'preg': preg, 'plas': plas, 'pres': pres, 'skin': skin, 
                'insu': insu, 'mass': mass, 'pedi': pedi, 'age': age
            }
            
            # Match input array mapping to trained column names if possible
            # Assuming standard order or passing a dataframe
            input_df = pd.DataFrame([features_dict])
            
            try:
                # If column names match exactly this works perfectly
                prediction_idx = model.predict(input_df)[0]
                probability = np.max(model.predict_proba(input_df)[0])
                
                result_class = target_names[prediction_idx]
                
                st.divider()
                if result_class == 'Positive':
                    st.warning(f"⚠️ **Diagnosis: {result_class.upper()}** (Confidence: {probability:.2%})")
                    st.markdown("Patient exhibits markers for diabetes. Recommend a clinical HbA1c test.")
                else:
                    st.success(f"✅ **Diagnosis: {result_class.upper()}** (Confidence: {probability:.2%})")
                    st.markdown("Patient tests negative for diabetes markers.")
            except Exception as e:
                st.error("Feature mapping failed (dataset version mismatch). Showing blind prediction.")
                # Fallback if pandas column alignment fails
                arr = np.array([[preg, plas, pres, skin, insu, mass, pedi, age]])
                prediction_idx = model.predict(arr)[0]
                result_class = target_names[prediction_idx]
                st.info(f"Result: {result_class}")

# ------------- Pneumonia Logic -------------
elif disease_type == "Pneumonia":
    st.header("Pneumonia Diagnosis (Medical Imaging)")
    st.markdown("Upload a Patient Chest X-Ray image (.png, .jpg) to run inference using a pre-trained ResNet-18 Deep Learning pipeline.")
    
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Chest X-Ray', use_container_width=True)
        
        if st.button("Diagnose Image via CNN"):
            with st.spinner("Processing image through Deep Learning pipeline..."):
                try:
                    result_class, confidence = predict_pneumonia(uploaded_file, pneumonia_model)
                    st.divider()
                    if result_class == "Pneumonia":
                        st.error(f"🚨 **Diagnosis: {result_class.upper()}** (Confidence: {confidence:.2%})")
                        st.markdown("Opaque areas indicative of pneumonia detected in lungs.")
                    else:
                        st.success(f"✅ **Diagnosis: {result_class.upper()}** (Confidence: {confidence:.2%})")
                        st.markdown("Lungs appear clear with no obvious signs of pneumonia infiltration.")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
