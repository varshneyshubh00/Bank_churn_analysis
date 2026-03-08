import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# ==========================================
# 1. PAGE SETUP & ASSET LOADING
# ==========================================
st.set_page_config(page_title="Bank Churn Predictor", layout="wide")
st.title("🏦 Customer Churn Risk Dashboard")
st.markdown("Predict customer churn, analyze risk factors, and simulate what-if scenarios.")

@st.cache_resource 
def load_assets():
    model = joblib.load('best_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, scaler, feature_cols

model, scaler, feature_cols = load_assets()

# ==========================================
# 2. HELPER FUNCTION: PREPROCESS UPLOADED CSV
# ==========================================
def preprocess_uploaded_data(df, feature_cols):
    df_clean = df.copy()
    
    cols_to_drop = [c for c in ['CustomerId', 'Surname', 'Exited'] if c in df_clean.columns]
    df_clean = df_clean.drop(cols_to_drop, axis=1)
    
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    df_clean.fillna('Unknown', inplace=True)
    
    df_clean = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=True)
    
    if 'Balance' in df_clean.columns and 'EstimatedSalary' in df_clean.columns:
        df_clean['Balance_to_Salary'] = df_clean['Balance'] / df_clean['EstimatedSalary']
        df_clean['Balance_to_Salary'].replace([np.inf, -np.inf], 0, inplace=True) 
        
    if 'NumOfProducts' in df_clean.columns and 'Tenure' in df_clean.columns:
        df_clean['Product_Density'] = df_clean['NumOfProducts'] / (df_clean['Tenure'] + 0.01)
        
    if 'IsActiveMember' in df_clean.columns and 'NumOfProducts' in df_clean.columns:
        df_clean['Engagement_Product'] = df_clean['IsActiveMember'] * df_clean['NumOfProducts']
        
    if 'Age' in df_clean.columns and 'Tenure' in df_clean.columns:
        df_clean['Age_Tenure'] = df_clean['Age'] * df_clean['Tenure']

    df_clean = df_clean.reindex(columns=feature_cols, fill_value=0)
    return df_clean

# ==========================================
# 3. BUILD THE TABS
# ==========================================
tab1, tab2 = st.tabs(["📂 Bulk CSV Upload & Analysis", "🎛️ What-If Scenario Simulator"])

# ------------------------------------------
# TAB 1: BULK CSV UPLOAD
# ------------------------------------------
with tab1:
    st.header("Bulk Prediction & Insights")
    uploaded_file = st.file_uploader("Upload your customer data (CSV format)", type=['csv'])
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(raw_df.head())
        
        processed_df = preprocess_uploaded_data(raw_df, feature_cols)
        scaled_data = scaler.transform(processed_df)
        
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)[:, 1]
        
        results_df = raw_df.copy()
        results_df['Churn_Probability'] = probabilities
        results_df['Predicted_Churn'] = predictions
        
        st.write("### Prediction Results:")
        st.dataframe(results_df[['CustomerId', 'Surname', 'Churn_Probability', 'Predicted_Churn']] if 'CustomerId' in results_df.columns else results_df)
        
        # Probability distribution visualization [cite: 42]
        st.write("### Probability Distribution Visualization")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probabilities, bins=20, color='coral', edgecolor='black')
        ax.set_xlabel('Churn Probability')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Distribution of Predicted Churn Probabilities')
        st.pyplot(fig)
        
        # Feature importance dashboard [cite: 43]
        st.write("### Feature Importance Dashboard")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_data)
        
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_to_plot = shap_values[:, :, 1]
        else:
            shap_values_to_plot = shap_values

        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_to_plot, processed_df, feature_names=feature_cols, show=False)
        st.pyplot(fig_shap)

# ------------------------------------------
# TAB 2: WHAT-IF SCENARIO SIMULATOR
# ------------------------------------------
with tab2:
    st.header("What-If Scenario Simulator")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("Input customer features")
        # Added Year and Credit Score!
        year = st.number_input("Year", min_value=2000, value=2025) 
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 100, 40)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
        salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0)
        
        st.markdown("**Adjust engagement/product values**")
        num_products = st.slider("Number of Products", 1, 4, 2)
        is_active = st.selectbox("Is Active Member?", [0, 1], index=1)
        has_crcard = st.selectbox("Has Credit Card?", [0, 1], index=1)
        
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col_result:
        st.subheader("Customer churn risk calculator")
        
        # Background math for the single simulator
        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0
        gender_male = 1 if gender == "Male" else 0
        balance_to_salary = balance / salary if salary > 0 else 0
        product_density = num_products / (tenure + 0.01)
        engagement_product = is_active * num_products
        age_tenure = age * tenure

        # Build the exact dataframe row using a DICTIONARY (Much safer!)
        sim_data_dict = {
            'Year': year,
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_crcard,
            'IsActiveMember': is_active,
            'EstimatedSalary': salary,
            'Geography_Germany': geo_germany,
            'Geography_Spain': geo_spain,
            'Gender_Male': gender_male,
            'Balance_to_Salary': balance_to_salary,
            'Product_Density': product_density,
            'Engagement_Product': engagement_product,
            'Age_Tenure': age_tenure
        }
        
        # Convert dictionary to DataFrame and force the exact column order the model expects
        sim_data = pd.DataFrame([sim_data_dict])
        sim_data = sim_data[feature_cols]

        # Scale and predict
        sim_scaled = scaler.transform(sim_data)
        sim_pred = model.predict(sim_scaled)[0]
        sim_prob = model.predict_proba(sim_scaled)[0][1] 
        
        # Observe churn probability changes
        st.metric(label="Calculated Churn Probability", value=f"{sim_prob * 100:.2f}%")
        
        if sim_pred == 1:
            st.error("🚨 HIGH RISK: Customer is predicted to CHURN.")
        else:
            st.success("✅ LOW RISK: Customer is predicted to RETAIN.")