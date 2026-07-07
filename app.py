import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import easyocr
from PIL import Image
import tempfile
import os

# -------------------------------
# Load saved models / mappings (using pickle)
# -------------------------------
with open("saved_models/category_model.pkl", "rb") as f:
    category_model = pickle.load(f)

with open("saved_models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("saved_models/category_encoder.pkl", "rb") as f:
    category_encoder = pickle.load(f)

with open("saved_models/issue_resolution_map.pkl", "rb") as f:
    issue_resolution_map = pickle.load(f)

with open("saved_models/category_issue_lookup.pkl", "rb") as f:
    category_issue_lookup = pickle.load(f)

# OCR reader
ocr_reader = easyocr.Reader(['en'], gpu=False)


# -------------------------------
# Prediction function
# -------------------------------
def predict_ticket(ticket_text):
    text_vec = tfidf.transform([ticket_text])

    pred_cat_label = category_model.predict(text_vec)[0]
    pred_category = category_encoder.inverse_transform([pred_cat_label])[0]

    candidate_df = category_issue_lookup[pred_category].copy()
    candidate_vectors = tfidf.transform(candidate_df["input_text"])

    sims = cosine_similarity(text_vec, candidate_vectors).flatten()
    best_idx = sims.argmax()

    pred_issue = candidate_df.iloc[best_idx]["issue_type"]
    pred_resolution = issue_resolution_map.get(pred_issue, "No solution found")

    return {
        "predicted_category": pred_category,
        "predicted_issue_type": pred_issue,
        "suggested_solution": pred_resolution
    }


# -------------------------------
# OCR function
# -------------------------------
def extract_text_from_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    result = ocr_reader.readtext(temp_path, detail=0)
    extracted_text = " ".join(result)

    os.remove(temp_path)
    return extracted_text


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="IT Ticket Auto-Resolution System", layout="wide")

st.title("🖥️ Intelligent IT Ticket Auto-Resolution System")
st.write("Automatically classify IT tickets and suggest instant solutions.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Enter Ticket Text")
    ticket_text = st.text_area("", height=200, placeholder="Describe your IT issue here...")

with col2:
    st.subheader("🖼️ Or Upload Screenshot")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

final_text = ticket_text

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Screenshot", use_container_width=True)
    with st.spinner("🔍 Extracting text from screenshot..."):
        image_text = extract_text_from_image(uploaded_file)
    st.subheader("📄 Extracted Text from Screenshot")
    st.write(image_text)

    if final_text.strip():
        final_text = final_text + " " + image_text
    else:
        final_text = image_text

if st.button("🔮 Predict Ticket", type="primary"):
    if not final_text.strip():
        st.warning("⚠️ Please enter ticket text or upload a screenshot.")
    else:
        with st.spinner("🤖 Predicting..."):
            result = predict_ticket(final_text)

        st.subheader("✅ Prediction Result")

        col3, col4, col5 = st.columns(3)

        with col3:
            st.success(f"**📂 Category:**\n{result['predicted_category']}")
        with col4:
            st.info(f"**🏷️ Issue Type:**\n{result['predicted_issue_type']}")
        with col5:
            st.warning(f"**💡 Solution:**\n{result['suggested_solution']}")