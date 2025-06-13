import streamlit as st
import joblib
import numpy as np
model=joblib.load('Yield_model.joblib')
le=joblib.load('crop_encoder.joblib')
st.title("🌾Crop Yield Prediction App")
crop=st.selectbox("Select Crop",le.classes_)
rain=st.number_input("Annual Rainfall(mm)",min_value=0.0,step=1.0)
fertilizer=st.number_input("fertilizer used(kg/ha)",min_value=0.0,step=1.0)
pesticide=st.number_input("Pesticide used(kg/ha)",min_value=0.0,step=1.0)

if st.button("Predict Yield"):
    try:
        crop_encoded=le.transform([crop])[0]
        features=np.array([[crop_encoded,rain,fertilizer,pesticide]])
        Predicted_yield=model.predict(features)[0]
        st.success(f"Predicted Yield: {Predicted_yield:.2f} tons/hectare")
    except Exception as e:
        st.error(f"Prediction Failed:{e}")


