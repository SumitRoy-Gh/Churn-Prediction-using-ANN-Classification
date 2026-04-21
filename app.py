import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1a2a6c, #b21f1f, #fdbb2d);
        background-attachment: fixed;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS AND ASSETS ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    return model, scaler, onehot_encoder_geo, label_encoder_gender

try:
    model, scaler, onehot_encoder_geo, label_encoder_gender = load_assets()
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# --- APP HEADER ---
st.title("🛡️ ChurnGuard AI")
st.subheader("Predicting Customer Retention with Deep Learning")
st.write("Fill in the customer details below to calculate the probability of churn.")

# --- USER INPUT SECTION ---
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 👤 Personal Profile")
        geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
        gender = st.selectbox("Gender", label_encoder_gender.classes_)
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)

    with col2:
        st.write("### 💰 Financial Profile")
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Annual Salary ($)", 0.0, 200000.0, 75000.0)

# --- PREDICTION LOGIC ---
if st.button("Calculate Churn Probability", use_container_width=True):
    # 1. Prepare base data point (excluding Geography for now)
    # Order: CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    
    # Label Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]
    
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # 2. One-Hot Encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # 3. Combine to match Trainer column order
    final_df = pd.concat([input_data, geo_encoded_df], axis=1)

    # 4. Scale inputs
    input_scaled = scaler.transform(final_df)

    # 5. Predict
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    # --- DISPLAY RESULTS ---
    st.divider()
    
    with st.container():
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.write("### Prediction Results")
        
        if prediction_proba > 0.5:
            st.error("⚠️ HIGH CHURN RISK")
            st.write(f"The model predicts the customer is likely to leave.")
        else:
            st.success("✅ LOW CHURN RISK")
            st.write(f"The model predicts the customer is likely to stay.")
        
        st.metric("Churn Probability", f"{prediction_proba:.2%}")
        st.progress(float(prediction_proba))
        st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.caption("Powered by TensorFlow and Streamlit • ChurnGuard AI v1.0")