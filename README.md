# 😊 EmotiSense AI: End-to-End Text Emotion Intelligence Web Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) and machine learning web application engineered to detect, classify, and analyze granular human emotional states in unstructured text in real time. Powered by TF-IDF vectorization and Multinomial Logistic Regression, **EmotiSense AI** provides interactive single-text inference, confidence probability breakdowns, enterprise batch CSV processing, and session-level audit logging via Streamlit.

---

## 📌 Project Overview

Understanding emotional undertones and fine-grained sentiment in customer feedback, support tickets, survey responses, and social media text is essential for actionable business intelligence, product strategy, and content moderation.

**EmotiSense AI** classifies text into 5 distinct emotional dimensions:
* 😃 **Joy / Happiness**
* 😢 **Sadness**
* 😠 **Anger**
* 😨 **Fear**
* 😲 **Surprise / Love** *(depending on training schema)*

---

## 🧠 Machine Learning Architecture & NLP Pipeline

The underlying NLP engine transforms raw, unstructured natural language into calibrated probability distributions:

```
Raw Input Text
      │
      ▼
[Text Preprocessing] ── Lowercasing, tokenization, whitespace cleanup
      │
      ▼
[TF-IDF Vectorizer] ── Term Frequency-Inverse Document Frequency matrix construction
      │
      ▼
[Multinomial Logistic Regression] ── Multi-class classification via Softmax decision boundary
      │
      ▼
[Calibrated Probabilities & Label Assignment]
```

### Key Technical Specs:
* **Feature Extraction:** Scikit-learn `TfidfVectorizer` (sublinear TF scaling, n-gram bounds)
* **Classifier:** Multinomial Logistic Regression (`L-BFGS` solver with balanced class weighting)
* **Serialization:** Python `joblib` / `pickle` pipeline (`model.pkl`, `vectorizer.pkl`)
* **Inference Latency:** $< 10\text{ ms}$ per sentence

---

## ✨ Features & Capabilities

* **⚡ Real-Time Sentiment & Emotion Inference:** Type or paste any phrase or paragraph to receive instant dominant emotion prediction with exact confidence scores.
* **📊 Interactive Plotly Probability Breakdown:** Interactive bar and radar charts displaying probability margins across all emotional classes.
* **📁 Enterprise Batch CSV Processing:** Upload bulk `.csv` datasets, automatically classify thousands of comments or reviews, visualize aggregate distribution charts, and download enriched data with proper `utf-8-sig` encoding.
* **📝 Live Session Audit Logs:** Track all prediction queries during the current session for verification and error analysis.

---

## 📂 Repository Structure

```plaintext
emotion_app/
├── .venv/                   # Python virtual environment
├── app.py                   # Main Streamlit dashboard application
├── emotion_model.pkl        # Serialized Multinomial Logistic Regression model
├── tfidf_vectorizer.pkl     # Serialized TF-IDF vectorizer artifact
├── test_batch.csv           # Sample test batch CSV for quick audit verification
├── requirements.txt         # Production dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/text-emotion-classifier.git
cd text-emotion-classifier
```

### 2. Set Up Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Ensure your `requirements.txt` includes: `streamlit`, `scikit-learn`, `pandas`, `numpy`, `plotly`, `joblib`)*

### 4. Run the Streamlit Application
```powershell
streamlit run app.py
```

The web dashboard will launch in your browser at `http://localhost:8501`.

---

## 🚀 Cloud Deployment (Streamlit Community Cloud)

1. Push your repository containing `app.py`, `emotion_model.pkl`, `tfidf_vectorizer.pkl`, and `requirements.txt` to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy!** — your app is live 24/7 with a public `.streamlit.app` URL.

---

## 🎓 Mentorship & Acknowledgments

This project was developed under the expert guidance and mentorship of **Mr. Avinash Pingale** as part of the specialized Machine Learning and Natural Language Processing training curriculum at **Skillio**.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
