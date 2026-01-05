import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config("Spam Email Streaming App", "📩")

st.title("📩 Spam Email Dataset Streaming App")

st.sidebar.title("Controls")

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Spam Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully ✅")``````

    # ---------------- DATA PREVIEW ----------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    # ---------------- COLUMN SELECTION ----------------
    text_col = st.selectbox("Select Text Column", df.columns)
    label_col = st.selectbox("Select Label Column", df.columns)

    # ---------------- TRAIN MODEL ----------------
    if st.button("🚀 Train Spam Model"):
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(df[text_col])
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = MultinomialNB()
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        st.success(f"Model Trained Successfully 🎉")
        st.info(f"Accuracy: {accuracy:.2f}")

        # ---------------- SAVE MODEL ----------------
        pickle.dump(model, open("spam_model.pkl", "wb"))
        pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

        st.success("Model Saved for Future Use 💾")

    # ---------------- LIVE PREDICTION ----------------
    st.subheader("Live Spam Check")

    email_text = st.text_area("Enter Email Text")

    if st.button("Check Spam"):
        model = pickle.load(open("spam_model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

        data = vectorizer.transform([email_text])
        pred = model.predict(data)[0]

        if pred == 1 or pred == "spam":
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam")

else:
    st.info("Upload a CSV file to start 🚀")
