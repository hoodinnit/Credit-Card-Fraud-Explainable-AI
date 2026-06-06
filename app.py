import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Fraud AI Dashboard", layout="centered")
st.title("💳 Credit Card Fraud Detection AI")
st.write("An end-to-end machine learning pipeline with SHAP explainability.")

# --- 2. ASSET LOADING ---
# The @st.cache_resource decorator prevents the app from reloading 
# the heavy models every time you click a button.
@st.cache_resource
def load_assets():
    model = joblib.load('fraud_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    X_demo = pd.read_csv('demo_features.csv')
    y_demo = pd.read_csv('demo_labels.csv')
    return model, explainer, X_demo, y_demo

model, explainer, X_demo, y_demo = load_assets()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Test the AI")
st.sidebar.write("Simulate a transaction from our test database:")

transaction_type = st.sidebar.radio("Select Transaction Type:", ["Normal", "Fraud"])

if st.sidebar.button("Load Transaction"):
    # Filter the demo data based on user selection
    target_class = 0 if transaction_type == "Normal" else 1
    subset = X_demo[y_demo['Class'] == target_class]
    
    # Pick a random transaction and save it to the session state
    random_tx = subset.sample(1)
    st.session_state['current_tx'] = random_tx

# --- 4. MAIN DASHBOARD ---
if 'current_tx' in st.session_state:
    tx = st.session_state['current_tx']
    
    st.subheader("Transaction Details")
    st.write(f"**Amount:** ${tx['Amount'].values[0]:.2f}")
    st.caption("*(The 28 anonymized PCA features are loaded in the background)*")
    
    if st.button("Run Risk Analysis", type="primary"):
        with st.spinner("Analyzing transaction vectors..."):
            # A. Make Prediction
            prob = model.predict_proba(tx)[0][1]
            pred = model.predict(tx)[0]
            
            # B. Display Result
            if pred == 1:
                st.error(f"🚨 FRAUD DETECTED! (Risk Score: {prob*100:.1f}%)")
            else:
                st.success(f"✅ APPROVED (Risk Score: {prob*100:.1f}%)")
                
            # C. Generate SHAP Explanation
            st.subheader("Why did the AI make this decision?")
            
            shap_values = explainer.shap_values(tx)
            
            # Safely handle SHAP version differences
            if isinstance(shap_values, list):
                fraud_shap = shap_values[1]
            else:
                fraud_shap = shap_values[:, :, 1]
                
            # Render the plot in Streamlit
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(fraud_shap, tx, plot_type="bar", show=False)
            st.pyplot(fig)