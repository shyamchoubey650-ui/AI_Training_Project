
# python -m streamlit run app.py  for running the code in terminal. n

import streamlit as st
import joblib

st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

# Load model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("📧 Smart Email Spam Detection System")
st.write("This system classifies emails as Spam, Not Spam, or Suspicious based on confidence levels.")

email_text = st.text_area("Enter Email Content", height=200)

if st.button("Predict"):

    if email_text.strip() == "":
        st.warning("⚠ Please enter email text.")
    else:
        vector = vectorizer.transform([email_text])
        probabilities = model.predict_proba(vector)

        spam_prob = probabilities[0][1]
        ham_prob = probabilities[0][0]

        st.markdown("---")

        # Decision Logic
        if spam_prob > 0.7:
            st.error("🚨 Spam Email Detected!")
        elif spam_prob < 0.3:
            st.success("✅ Not Spam (Safe Email)")
        else:
            st.warning("⚠ Suspicious Email (Low Confidence Prediction)")

        # Show probabilities
        st.write("### 📊 Prediction Confidence")
        st.write(f"Spam Probability: *{spam_prob:.4f}*")
        st.write(f"Ham Probability: *{ham_prob:.4f}*")

        # Visual bars
        st.write("Spam Confidence")
        st.progress(float(spam_prob))

        st.write("Ham Confidence")
        st.progress(float(ham_prob))