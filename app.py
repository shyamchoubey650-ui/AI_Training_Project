
# python -m streamlit run app.py  for running the code in terminal. 

import streamlit as st
import joblib

# Page settings
st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Title
st.title("📧 Email Spam Detection System")
st.write("Enter the email content below to check whether it is Spam or Not Spam.")

# Text area
email_text = st.text_area("Enter Email Content", height=200)

# Predict button
if st.button("Predict"):

    if email_text.strip() == "":
        st.warning("⚠ Please enter email text.")
    else:
        # Transform input
        vector = vectorizer.transform([email_text])

        # Prediction
        prediction = model.predict(vector)
        probabilities = model.predict_proba(vector)

        spam_prob = probabilities[0][1]
        ham_prob = probabilities[0][0]

        st.markdown("---")

        # Result
        if prediction[0] == 1:
            st.error("🚨 Spam Email Detected!")
        else:
            st.success("✅ Not Spam (Safe Email)")

        # Show probabilities
        st.write("### 📊 Prediction Confidence")
        st.write(f"Spam Probability: **{spam_prob:.4f}**")
        st.write(f"Ham Probability: **{ham_prob:.4f}**")

        # Progress bars
        st.progress(float(spam_prob))
        st.progress(float(ham_prob))