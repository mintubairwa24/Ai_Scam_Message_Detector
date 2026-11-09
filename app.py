import io
import re
import urllib.request
import zipfile

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI Scam Message Detector", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main {background: #f7f9fc;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#101f3d,#245b87); color: white; margin-bottom: 1rem;}
.hero h1 {color: white; margin-bottom: .3rem;}
.hero p {color: #e7f0fa; margin-bottom: 0;}
.small-note {color: #667085; font-size: .88rem;}
.result-safe {padding: 1rem; border-radius: 12px; background: #e8f7ee; border: 1px solid #a7dfba;}
.result-risk {padding: 1rem; border-radius: 12px; background: #fff0ed; border: 1px solid #f2b7aa;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_sms_dataset():
    """Download and read the UCI SMS Spam Collection without ucimlrepo's API."""
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SMSScamDetector/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        dataset_file = next(
            (name for name in archive.namelist() if name.rsplit("/", 1)[-1] == "SMSSpamCollection"),
            None,
        )
        if dataset_file is None:
            raise ValueError("The UCI download did not contain SMSSpamCollection.")
        with archive.open(dataset_file) as dataset_stream:
            data = pd.read_csv(
                dataset_stream,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8",
                on_bad_lines="skip",
            )

    data = data.dropna(subset=["label", "message"])
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str)
    data = data[data["label"].isin(["ham", "spam"])]
    if data.empty:
        raise ValueError("The downloaded UCI dataset contains no usable records.")
    return data


@st.cache_resource(show_spinner="Downloading the public SMS dataset and training the model...")
def train_model():
    dataset = load_sms_dataset()
    X = dataset["message"]
    y = dataset["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("classifier", MultinomialNB(alpha=0.35)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "test_size": len(y_test),
        "dataset_size": len(dataset),
    }
    return model, metrics


st.markdown("""
<div class="hero">
  <h1>🛡️ AI Scam &amp; Phishing Message Detector</h1>
  <p>Check suspicious SMS and short messages with a machine-learning model.</p>
</div>
""", unsafe_allow_html=True)

st.write("Paste a message below. The model estimates whether it resembles spam or a legitimate message and highlights warning signs.")

with st.sidebar:
    st.header("About this project")
    st.write("**Problem:** Fraudulent messages can trick people into sharing personal details, clicking unsafe links, or sending money.")
    st.write("**AI approach:** TF-IDF text features + Multinomial Naive Bayes classifier.")
    st.write("**Training data:** UCI SMS Spam Collection (5,574 English messages).")
    st.caption("Educational prototype. It is not a bank, telecom, or government fraud-verification service.")

try:
    model, metrics = train_model()
except Exception as e:
    st.error("The model could not load its public training dataset. Please refresh the app and try again.")
    st.caption("Technical detail: " + str(e)[:350])
    st.stop()

left, right = st.columns([1.55, 1])
with left:
    message = st.text_area(
        "Enter or paste a suspicious message",
        height=180,
        placeholder="Example: Your account will be blocked today. Verify your details at http://example.com",
    )
    analyze = st.button("Analyze message", type="primary", use_container_width=True)

with right:
    st.subheader("Model test results")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    c2.metric("F1 score", f"{metrics['f1'] * 100:.1f}%")
    c3, c4 = st.columns(2)
    c3.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    c4.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    st.caption(f"Evaluated on a held-out test set of {metrics['test_size']:,} messages from {metrics['dataset_size']:,} records.")

warning_patterns = {
    "Urgent pressure": r"\b(urgent|immediately|today|now|expire|suspend|blocked|verify immediately)\b",
    "Requests sensitive information": r"\b(password|pin|otp|cvv|card number|account details|aadhaar|pan)\b",
    "Prize or reward claim": r"\b(won|winner|prize|lottery|claim|reward|free gift)\b",
    "Payment request": r"\b(pay|payment|transfer|fee|deposit|upi|refund)\b",
    "Link included": r"(https?://|www\.)",
}

if analyze:
    if len(message.strip()) < 5:
        st.warning("Please enter a longer message so the model has enough text to analyze.")
    else:
        prediction = int(model.predict([message])[0])
        probabilities = model.predict_proba([message])[0]
        spam_probability = float(probabilities[1])
        matched = [name for name, pattern in warning_patterns.items() if re.search(pattern, message, re.I)]

        st.divider()
        st.subheader("Analysis result")
        if prediction == 1:
            st.markdown(f"""
            <div class="result-risk">
              <h3>⚠️ Potential spam / scam-like message</h3>
              <p>Model spam score: <b>{spam_probability * 100:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
              <h3>✅ Likely legitimate message</h3>
              <p>Model spam score: <b>{spam_probability * 100:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)

        st.progress(spam_probability, text="Estimated spam probability")
        st.write("**Potential warning signs found in the text:**")
        if matched:
            for item in matched:
                st.write(f"- {item}")
        else:
            st.write("- No obvious warning phrase from the simple rule-based checklist was found.")

        st.info("The warning-sign checklist is separate from the ML prediction. A message can still be fraudulent even if no warning phrase is detected.")
        st.subheader("What should you do?")
        st.write("- Do not share OTPs, PINs, passwords, CVV numbers, or identity details.")
        st.write("- Do not click unexpected links or install apps from message links.")
        st.write("- Verify requests using the official app, website, or phone number you find independently.")
        st.write("- If money is lost in India, contact your bank immediately and report cyber fraud through 1930 or cybercrime.gov.in.")

st.divider()
with st.expander("How does the AI model work?"):
    st.write("1. The text is converted into numerical TF-IDF features.")
    st.write("2. A Multinomial Naive Bayes classifier estimates whether the message resembles spam or legitimate examples.")
    st.write("3. The app displays the predicted class and estimated spam probability.")
    st.write("4. A separate keyword checklist highlights common warning signs.")
    st.write("The model is trained at app startup using the public UCI SMS Spam Collection dataset.")

st.caption("Dataset: Almeida & Hidalgo (2011), UCI Machine Learning Repository, SMS Spam Collection, DOI: 10.24432/C5CC84. The dataset is English-language and may not generalize to Hindi, Hinglish, regional languages, or newer scam tactics.")
st.caption("Privacy: This demo does not save messages to a database. Avoid entering real OTPs, account details, phone numbers, or other sensitive information.")
