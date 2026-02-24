import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Import the PyTorch module we built
from image_inference import pneumonia_model, predict_pneumonia, pancreatitis_model, predict_pancreatitis

st.set_page_config(page_title="Disease Diagnosis System", page_icon="⚕️", layout="wide")

st.title("⚕️ Multi-Disease Diagnosis System")
st.markdown("Use this system to diagnose Diabetes, Breast Cancer, or Pneumonia using patient records or medical imaging.")

# Sidebar navigation
disease_type = st.sidebar.selectbox(
    "Select Disease to Diagnose",
    ["Breast Cancer", "Diabetes", "Pneumonia", "Pancreatitis"]
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
        target_names = model_data["target_names"]
        scaler = model_data.get("scaler", None)
        
        st.subheader("Patient Vitals")
        st.caption("Adjust the values below to match the patient's medical record. Normal reference ranges are shown.")
        col1, col2 = st.columns(2)
        
        preg = col1.number_input("🤰 Pregnancies", 0, 20, 1, help="Number of times pregnant")
        plas = col2.slider("🩸 Glucose (mg/dL)  [Normal: 70–140]", 0, 200, 120)
        pres = col1.slider("💓 Blood Pressure (mm Hg)  [Normal: 60–80]", 0, 140, 72)
        skin = col2.slider("📏 Skin Thickness (mm)  [Normal: 10–50]", 0, 100, 29)
        insu = col1.slider("💉 Insulin (mu U/ml)  [Normal: 16–166]", 0, 900, 125)
        mass = col2.slider("⚖️ BMI  [Normal: 18.5–24.9]", 0.0, 70.0, 32.0, 0.1)
        pedi = col1.slider("🧬 Diabetes Pedigree Function  [Normal: 0.0–1.0]", 0.0, 2.5, 0.47, 0.01)
        age = col2.number_input("🎂 Age", 1, 120, 33)

        if st.button("🔍 Diagnose Diabetes"):
            features_dict = {
                'preg': float(preg), 'plas': float(plas), 'pres': float(pres), 
                'skin': float(skin), 'insu': float(insu), 'mass': float(mass), 
                'pedi': float(pedi), 'age': float(age)
            }
            
            input_df = pd.DataFrame([features_dict])
            
            # Apply the scaler if available (matches the training pipeline)
            if scaler is not None:
                input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
            else:
                input_scaled = input_df
            
            prediction_idx = int(model.predict(input_scaled)[0])
            proba = model.predict_proba(input_scaled)[0]
            probability = float(np.max(proba))
            positive_prob = float(proba[1])
            
            result_class = target_names[prediction_idx]
            
            st.divider()
            if result_class == 'Positive':
                st.warning(f"⚠️ **Diagnosis: {result_class.upper()}** (Confidence: {probability:.2%})")
                st.markdown("Patient exhibits markers for diabetes. Recommend a clinical HbA1c test.")
            else:
                st.success(f"✅ **Diagnosis: {result_class.upper()}** (Confidence: {probability:.2%})")
                st.markdown("Patient tests negative for diabetes markers.")
            
            # Risk factor breakdown
            st.subheader("📊 Risk Factor Analysis")
            risk_data = {
                "Factor": ["Glucose", "BMI", "Age", "Blood Pressure", "Insulin", "Pedigree"],
                "Value": [plas, mass, age, pres, insu, pedi],
                "Normal Range": ["70–140 mg/dL", "18.5–24.9", "< 45", "60–80 mm Hg", "16–166 mu U/ml", "0.0–1.0"],
                "Status": [
                    "🔴 High" if plas > 140 else ("🟡 Borderline" if plas > 120 else "🟢 Normal"),
                    "🔴 Obese" if mass > 30 else ("🟡 Overweight" if mass > 25 else "🟢 Normal"),
                    "🟡 Risk factor" if age > 45 else "🟢 Normal",
                    "🔴 High" if pres > 90 else ("🟡 Elevated" if pres > 80 else "🟢 Normal"),
                    "🔴 High" if insu > 166 else ("🟡 Low" if insu < 16 else "🟢 Normal"),
                    "🔴 High" if pedi > 1.0 else ("🟡 Moderate" if pedi > 0.5 else "🟢 Normal"),
                ]
            }
            st.table(pd.DataFrame(risk_data))
            
            # Show diabetes probability gauge
            st.metric("Diabetes Probability", f"{positive_prob:.1%}")

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

# ------------- Pancreatitis Logic -------------
elif disease_type == "Pancreatitis":
    st.header("Pancreatitis Diagnosis (CT Scan Imaging)")
    st.markdown("Upload a patient abdominal **CT scan** image to detect signs of pancreatitis using a pre-trained DenseNet-121 Deep Learning pipeline.")
    
    st.info("💡 **Tip:** For best results, upload axial CT scan slices focused on the pancreatic region. Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`")
    
    uploaded_ct = st.file_uploader("Upload CT Scan Image", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="ct_upload")
    
    if uploaded_ct is not None:
        image = Image.open(uploaded_ct)
        st.image(image, caption='Uploaded CT Scan', use_container_width=True)
        
        if st.button("Diagnose CT Scan via CNN"):
            with st.spinner("Analyzing CT scan through DenseNet-121 pipeline..."):
                try:
                    result_class, confidence, grad_cam_fig = predict_pancreatitis(uploaded_ct, pancreatitis_model)
                    st.divider()
                    if result_class == "Pancreatitis":
                        st.error(f"🚨 **Diagnosis: {result_class.upper()}** (Confidence: {confidence:.2%})")
                        st.markdown("Abnormal pancreatic tissue patterns detected. Indicators consistent with inflammation of the pancreas.")
                        st.markdown("**Recommended next steps:** Serum lipase/amylase test, contrast-enhanced CT, clinical correlation.")
                    else:
                        st.success(f"✅ **Diagnosis: {result_class.upper()}** (Confidence: {confidence:.2%})")
                        st.markdown("Pancreatic region appears within normal limits. No obvious signs of acute or chronic pancreatitis.")
                    
                    # Display the Grad-CAM analysis figure
                    st.subheader("🔬 Grad-CAM Pattern Analysis")
                    st.markdown("The heatmap below highlights the regions of the CT scan that the CNN focused on to make its prediction. **Red/yellow** regions indicate high activation (areas of interest), while **blue** regions had minimal influence.")
                    st.pyplot(grad_cam_fig)
                except Exception as e:
                    st.error(f"Error processing CT scan: {str(e)}")
