import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained pipeline
# -------------------------------
pipeline = joblib.load("drug_pipeline.pkl")

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Drug Regulatory Risk Assessment", layout="wide")

st.title("💊 Drug Regulatory Risk Assessment System")
st.markdown("### Supervised Machine Learning Classification Model")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Drug Details")

Dosage_mg = st.sidebar.number_input("Dosage (mg)", value=100)
Price_Per_Unit = st.sidebar.number_input("Price Per Unit", value=10.0)
Production_Cost = st.sidebar.number_input("Production Cost", value=5.0)
Marketing_Spend = st.sidebar.number_input("Marketing Spend", value=1000.0)
Clinical_Trial_Phase = st.sidebar.selectbox("Clinical Trial Phase", [1, 2, 3, 4])
Side_Effect_Severity_Score = st.sidebar.slider("Side Effect Severity Score", 0.0, 10.0, 5.0)
Abuse_Potential_Score = st.sidebar.slider("Abuse Potential Score", 0.0, 10.0, 3.0)
Prescription_Rate = st.sidebar.number_input("Prescription Rate", value=50.0)
Hospital_Distribution_Percentage = st.sidebar.number_input("Hospital Distribution (%)", value=40.0)
Pharmacy_Distribution_Percentage = st.sidebar.number_input("Pharmacy Distribution (%)", value=60.0)
Annual_Sales_Volume = st.sidebar.number_input("Annual Sales Volume", value=10000.0)
Regulatory_Risk_Score = st.sidebar.slider("Regulatory Risk Score", 0.0, 10.0, 4.0)
Approval_Time_Months = st.sidebar.number_input("Approval Time (Months)", value=12)
Patent_Duration_Years = st.sidebar.number_input("Patent Duration (Years)", value=10)
RD_Investment_Million = st.sidebar.number_input("R&D Investment (Million)", value=50.0)
Competitor_Count = st.sidebar.number_input("Competitor Count", value=5)
Recall_History_Count = st.sidebar.number_input("Recall History Count", value=0)
Adverse_Event_Reports = st.sidebar.number_input("Adverse Event Reports", value=10)
Insurance_Coverage_Percentage = st.sidebar.number_input("Insurance Coverage (%)", value=70.0)
Export_Percentage = st.sidebar.number_input("Export Percentage (%)", value=30.0)
Online_Sales_Percentage = st.sidebar.number_input("Online Sales (%)", value=20.0)
Brand_Reputation_Score = st.sidebar.slider("Brand Reputation Score", 0.0, 10.0, 8.0)
Doctor_Recommendation_Rate = st.sidebar.number_input("Doctor Recommendation Rate", value=75.0)

Drug_Form = st.sidebar.selectbox("Drug Form", ["Tablet", "Capsule", "Injection", "Syrup"])
Therapeutic_Class = st.sidebar.selectbox("Therapeutic Class", ["Antibiotic", "Analgesic", "Antiviral", "Antifungal"])
Manufacturing_Region = st.sidebar.selectbox("Manufacturing Region", ["Asia", "Europe", "USA"])
Requires_Cold_Storage = st.sidebar.selectbox("Requires Cold Storage", ["Yes", "No"])
OTC_Flag = st.sidebar.selectbox("OTC Flag", ["Yes", "No"])
High_Risk_Substance = st.sidebar.selectbox("High Risk Substance", ["Yes", "No"])

# -------------------------------
# Main Result Section
# -------------------------------
st.subheader("🔎 Risk Assessment Result")

if st.sidebar.button("🚀 Run Risk Assessment"):

    input_df = pd.DataFrame([{
        "Dosage_mg": Dosage_mg,
        "Price_Per_Unit": Price_Per_Unit,
        "Production_Cost": Production_Cost,
        "Marketing_Spend": Marketing_Spend,
        "Clinical_Trial_Phase": Clinical_Trial_Phase,
        "Side_Effect_Severity_Score": Side_Effect_Severity_Score,
        "Abuse_Potential_Score": Abuse_Potential_Score,
        "Prescription_Rate": Prescription_Rate,
        "Hospital_Distribution_Percentage": Hospital_Distribution_Percentage,
        "Pharmacy_Distribution_Percentage": Pharmacy_Distribution_Percentage,
        "Annual_Sales_Volume": Annual_Sales_Volume,
        "Regulatory_Risk_Score": Regulatory_Risk_Score,
        "Approval_Time_Months": Approval_Time_Months,
        "Patent_Duration_Years": Patent_Duration_Years,
        "R&D_Investment_Million": RD_Investment_Million,
        "Competitor_Count": Competitor_Count,
        "Recall_History_Count": Recall_History_Count,
        "Adverse_Event_Reports": Adverse_Event_Reports,
        "Drug_Form": Drug_Form,
        "Therapeutic_Class": Therapeutic_Class,
        "Manufacturing_Region": Manufacturing_Region,
        "Requires_Cold_Storage": Requires_Cold_Storage,
        "OTC_Flag": OTC_Flag,
        "High_Risk_Substance": High_Risk_Substance,
        "Insurance_Coverage_Percentage": Insurance_Coverage_Percentage,
        "Export_Percentage": Export_Percentage,
        "Online_Sales_Percentage": Online_Sales_Percentage,
        "Brand_Reputation_Score": Brand_Reputation_Score,
        "Doctor_Recommendation_Rate": Doctor_Recommendation_Rate
    }])

    prediction = pipeline.predict(input_df)
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error("⚠️ High Regulatory Risk – Likely Regulated Drug")
    else:
        st.success("✅ Low Regulatory Risk – Likely Non-Regulated Drug")

    st.metric("Regulatory Risk Probability", f"{probability*100:.1f}%")

else:
    st.info("Fill in the drug parameters on the left and click 'Run Risk Assessment'.")