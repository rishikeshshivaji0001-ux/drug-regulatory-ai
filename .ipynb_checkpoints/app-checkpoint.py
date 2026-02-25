import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("drug_regulatory_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Drug Regulatory Classification System")
st.write("Enter drug parameters below:")

# -------- Numeric Inputs --------

Dosage_mg = st.number_input("Dosage (mg)", value=100)
Price_Per_Unit = st.number_input("Price Per Unit", value=10.0)
Production_Cost = st.number_input("Production Cost", value=5.0)
Marketing_Spend = st.number_input("Marketing Spend", value=1000.0)
Clinical_Trial_Phase = st.number_input("Clinical Trial Phase", value=2)
Side_Effect_Severity_Score = st.number_input("Side Effect Severity Score", value=5.0)
Abuse_Potential_Score = st.number_input("Abuse Potential Score", value=3.0)
Prescription_Rate = st.number_input("Prescription Rate", value=50.0)
Hospital_Distribution_Percentage = st.number_input("Hospital Distribution (%)", value=40.0)
Pharmacy_Distribution_Percentage = st.number_input("Pharmacy Distribution (%)", value=60.0)
Annual_Sales_Volume = st.number_input("Annual Sales Volume", value=10000.0)
Regulatory_Risk_Score = st.number_input("Regulatory Risk Score", value=4.0)
Approval_Time_Months = st.number_input("Approval Time (Months)", value=12)
Patent_Duration_Years = st.number_input("Patent Duration (Years)", value=10)
RD_Investment_Million = st.number_input("R&D Investment (Million)", value=50.0)
Competitor_Count = st.number_input("Competitor Count", value=5)
Recall_History_Count = st.number_input("Recall History Count", value=0)
Adverse_Event_Reports = st.number_input("Adverse Event Reports", value=10)
Insurance_Coverage_Percentage = st.number_input("Insurance Coverage (%)", value=70.0)
Export_Percentage = st.number_input("Export Percentage (%)", value=30.0)
Online_Sales_Percentage = st.number_input("Online Sales (%)", value=20.0)
Brand_Reputation_Score = st.number_input("Brand Reputation Score", value=8.0)
Doctor_Recommendation_Rate = st.number_input("Doctor Recommendation Rate", value=75.0)

# -------- Encoded Categorical Placeholders --------
# (Since we used LabelEncoder during training)

Drug_Form = st.number_input("Drug Form (Encoded Number)", value=0)
Therapeutic_Class = st.number_input("Therapeutic Class (Encoded Number)", value=0)
Manufacturing_Region = st.number_input("Manufacturing Region (Encoded Number)", value=0)
Requires_Cold_Storage = st.number_input("Requires Cold Storage (0 or 1)", value=0)
OTC_Flag = st.number_input("OTC Flag (0 or 1)", value=0)
High_Risk_Substance = st.number_input("High Risk Substance (0 or 1)", value=0)

# -------- Prediction Button --------

if st.button("Predict"):

    input_data = np.array([[
        Dosage_mg,
        Price_Per_Unit,
        Production_Cost,
        Marketing_Spend,
        Clinical_Trial_Phase,
        Side_Effect_Severity_Score,
        Abuse_Potential_Score,
        Prescription_Rate,
        Hospital_Distribution_Percentage,
        Pharmacy_Distribution_Percentage,
        Annual_Sales_Volume,
        Regulatory_Risk_Score,
        Approval_Time_Months,
        Patent_Duration_Years,
        RD_Investment_Million,
        Competitor_Count,
        Recall_History_Count,
        Adverse_Event_Reports,
        Drug_Form,
        Therapeutic_Class,
        Manufacturing_Region,
        Requires_Cold_Storage,
        OTC_Flag,
        High_Risk_Substance,
        Insurance_Coverage_Percentage,
        Export_Percentage,
        Online_Sales_Percentage,
        Brand_Reputation_Score,
        Doctor_Recommendation_Rate
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ This Drug is Likely REGULATED")
    else:
        st.success("✅ This Drug is Likely NON-REGULATED")