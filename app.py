import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Payment Prediction 💳",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: white;
    }
    .stSelectbox, .stNumberInput {
        background-color: #2E2E2E;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load all pickle files
@st.cache_resource
def load_models_and_transformers():
    try:
        # Define paths
        PICKLE_PATH = "pickle"
        MODELS_PATH = os.path.join(PICKLE_PATH, "models")
        
        # Load encoders
        with open(os.path.join(PICKLE_PATH, 'education_level_OE.pkl'), 'rb') as f:
            education_level_encoder = pickle.load(f)
        with open(os.path.join(PICKLE_PATH, 'family_status_OE.pkl'), 'rb') as f:
            family_status_encoder = pickle.load(f)
        with open(os.path.join(PICKLE_PATH, 'gender_BE.pkl'), 'rb') as f:
            gender_encoder = pickle.load(f)
        with open(os.path.join(PICKLE_PATH, 'housing_type_OE.pkl'), 'rb') as f:
            housing_type_encoder = pickle.load(f)
        with open(os.path.join(PICKLE_PATH, 'income_type_OE.pkl'), 'rb') as f:
            income_type_encoder = pickle.load(f)
        with open(os.path.join(PICKLE_PATH, 'occupation_OE.pkl'), 'rb') as f:
            occupation_encoder = pickle.load(f)
        with open(os.path.join(PICKLE_PATH, 'owns_car_BE.pkl'), 'rb') as f:
            owns_car_encoder = pickle.load(f)
        with open(os.path.join(PICKLE_PATH, 'owns_realty_BE.pkl'), 'rb') as f:
            owns_realty_encoder = pickle.load(f)
            
        encoders = {
            'education_level': education_level_encoder,
            'family_status': family_status_encoder,
            'gender': gender_encoder,
            'housing_type': housing_type_encoder,
            'income_type': income_type_encoder,
            'occupation': occupation_encoder,
            'owns_car': owns_car_encoder,
            'owns_realty': owns_realty_encoder
        }
        
        # Load outlier transformers
        with open(os.path.join(PICKLE_PATH, 'months_employed_transform_metadata.pkl'), 'rb') as f:
            months_employed_transformer = pickle.load(f)
        with open(os.path.join(PICKLE_PATH, 'total_income_transform_metadata.pkl'), 'rb') as f:
            total_income_transformer = pickle.load(f)
            
        outlier_transformers = {
            'months_employed': months_employed_transformer,
            'total_income': total_income_transformer
        }
        
        # Load feature selector and scaler
        with open(os.path.join(PICKLE_PATH, 'selected_features.pkl'), 'rb') as f:
            selected_features = pickle.load(f)
        with open(os.path.join(PICKLE_PATH, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # Load ML models
        models = {
            'XGBoost 🚀': pickle.load(open(os.path.join(MODELS_PATH, 'XGBoost_model.pkl'), 'rb')),
            'LightGBM 💫': pickle.load(open(os.path.join(MODELS_PATH, 'lgb_model.pkl'), 'rb')),
            'Random Forest 🌳': pickle.load(open(os.path.join(MODELS_PATH, 'RF_model.pkl'), 'rb')),
            'CatBoost 😺': pickle.load(open(os.path.join(MODELS_PATH, 'catboost_model.pkl'), 'rb')),
            'Deep Learning 🧠': load_model(os.path.join(MODELS_PATH, 'my_model.keras'))
        }
        
        return encoders, outlier_transformers, selected_features, scaler, models
    
    except Exception as e:
        st.error(f"Error loading models and transformers: {str(e)}")
        return None, None, None, None, None

# Load all models and transformers
encoders, outlier_transformers, selected_features, scaler, models = load_models_and_transformers()

# Main app header
st.title("Credit Payment Prediction System 💳")
st.markdown("### Predict Customer Payment Reliability 🎯")
st.write("Enter customer information below to predict payment probability 📊")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Personal Information")
    gender = st.selectbox('Gender 👥', ['Male', 'Female'])
    owns_car = st.selectbox('Owns Car 🚗', ['Yes', 'No'])
    owns_realty = st.selectbox('Owns Realty 🏠', ['Yes', 'No'])
    total_income = st.number_input('Total Income 💰', min_value=0.0)
    income_type = st.selectbox('Income Type 💼', ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'])
    education_level = st.selectbox('Education Level 🎓', ['Secondary', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'])

with col2:
    st.markdown("#### 📋 Additional Information")
    family_status = st.selectbox('Family Status 💑', ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'])
    occupation = st.selectbox('Occupation 💼', ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff', 'Cleaning staff', 'Cooking staff', 'Low-skill Laborers', 'Private service staff', 'Waiters/barmen staff', 'HR staff', 'IT staff', 'Realty agents'])
    family_size = st.number_input('Family Size 👨‍👩‍👧‍👦', min_value=1, max_value=20)
    age = st.number_input('Age 📅', min_value=18, max_value=100)
    months_employed = st.number_input('Months Employed 📆', min_value=0)
    Months_sub = st.number_input('Months of Subscription 📅', min_value=0)

# Model selection
st.markdown("#### 🤖 Model Selection")
selected_model = st.selectbox('Select Prediction Model', list(models.keys()))

def preprocess_input(input_data):
    try:
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Handle binary encodings (stored as dictionaries)
        df['owns_car'] = df['owns_car'].map(encoders['owns_car'])
        df['owns_realty'] = df['owns_realty'].map(encoders['owns_realty'])
        
        # Handle OrdinalEncoder encodings
        df['income_type'] = encoders['income_type'].transform(df[['income_type']])[0]
        df['education_level'] = encoders['education_level'].transform(df[['education_level']])[0]
        df['family_status'] = encoders['family_status'].transform(df[['family_status']])[0]
        df['occupation'] = encoders['occupation'].transform(df[['occupation']])[0]
        
        # Handle numeric features
        try:
            df['total_income'] = outlier_transformers['total_income'].transform(df[['total_income']])
            df['months_employed'] = outlier_transformers['months_employed'].transform(df[['months_employed']])
        except:
            # If transformation fails, keep original values
            pass
        
        # Select only the required features
        df = df[selected_features]
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Scale the features
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)
        
        return df
    
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        st.error(f"Input data: {input_data}")
        
        # Add more detailed error information
        st.error("\nFeature values before processing:")
        for col in df.columns:
            st.error(f"{col}: {df[col].values}")
        return None

# Add this debugging information at the start of your app
st.sidebar.markdown("### 🔍 Debug Information")
if st.sidebar.checkbox("Show encoder information"):
    st.sidebar.write("Binary Encoders (Dictionaries):")
    st.sidebar.write("owns_car:", encoders['owns_car'])
    st.sidebar.write("owns_realty:", encoders['owns_realty'])
    
    st.sidebar.write("\nOrdinal Encoders:")
    for name, encoder in encoders.items():
        if isinstance(encoder, dict):
            continue
        st.sidebar.write(f"{name}: {type(encoder)}")
        if hasattr(encoder, 'categories_'):
            st.sidebar.write(f"Categories: {encoder.categories_}")

if st.button('Predict Payment Probability 🔮'):
    with st.spinner('Analyzing customer data... ⚙️'):
        try:
            # Prepare input data
            input_data = {
                'owns_car': owns_car,
                'owns_realty': owns_realty,
                'total_income': total_income,
                'income_type': income_type,
                'education_level': education_level,
                'family_status': family_status,
                'occupation': occupation,
                'family_size': family_size,
                'age': age,
                'months_employed': months_employed,
                'Months_sub': Months_sub
            }
            
            # Preprocess input
            processed_input = preprocess_input(input_data)
            
            if processed_input is not None:
                # Make prediction
                model = models[selected_model]
                if 'Deep Learning' in selected_model:
                    prediction = model.predict(processed_input)[0][0]
                else:
                    prediction = model.predict_proba(processed_input)[0][1]
                
                # Display results
                st.markdown("---")
                
                # Create three columns for centered display
                col1, col2, col3 = st.columns([1,2,1])
                
                with col2:
                    st.markdown("### 📊 Prediction Results")
                    
                    # Display probability
                    st.metric(
                        label="Payment Probability",
                        value=f"{prediction:.2%}",
                        delta="Will Pay ✅" if prediction > 0.5 else "Risk of Default ⚠️"
                    )
                    
                    # Create a colored box based on prediction
                    if prediction > 0.7:
                        st.success("""
                            🟢 Highly Reliable Customer
                            - Strong payment probability
                            - Low risk profile
                            - Recommended for approval
                        """)
                    elif prediction > 0.5:
                        st.warning("""
                            🟡 Moderate Risk Customer
                            - Acceptable payment probability
                            - Some risk factors present
                            - Additional verification recommended
                        """)
                    else:
                        st.error("""
                            🔴 High Risk Customer
                            - Low payment probability
                            - Significant risk factors
                            - Careful evaluation needed
                        """)
                    
                    # Display model confidence
                    st.markdown("#### 🎯 Prediction Details")
                    st.write(f"""
                    - Model Used: {selected_model}
                    - Confidence Score: {prediction:.2%}
                    - Status: {"Recommended ✅" if prediction > 0.5 else "Not Recommended ❌"}
                    """)
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Add informative footer
st.markdown('---')
st.markdown('### 📊 Credit Assessment System Information')

# Create three columns for footer information
fcol1, fcol2, fcol3 = st.columns(3)

with fcol1:
    st.markdown("""
    #### 🎯 Prediction Guide
    - Above 70%: High Reliability
    - 50-70%: Moderate Reliability
    - Below 50%: High Risk
    """)

with fcol2:
    st.markdown("""
    #### 💡 Model Features
    - Advanced ML Algorithms
    - Historical Data Analysis
    - Real-time Processing
    """)

with fcol3:
    st.markdown("""
    #### 🔒 System Benefits
    - Quick Assessment
    - Consistent Evaluation
    - Data-driven Decisions
    """)

# Final footer
st.markdown('---')
st.markdown("""
<div style='text-align: center'>
    <p>🏦 Powered by Advanced Machine Learning | Version 1.0</p>
    <p>Created with ❤️ by Your Name</p>
</div>
""", unsafe_allow_html=True)