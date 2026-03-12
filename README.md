# 🧠 EmoSense — Emotion Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-88%25-brightgreen?style=for-the-badge"/>
</p>

A machine learning web app that detects **6 human emotions** from text using Logistic Regression and TF-IDF vectorization — built with a sleek dark-themed Streamlit UI.

---

## 🎭 Emotions Detected

| Emotion | Label | Examples |
|---------|-------|---------|
| 😢 | Sadness | *"I can't stop crying, everything feels hopeless"* |
| 😠 | Anger | *"I am absolutely furious, how dare they!"* |
| ❤️ | Love | *"I feel so loved and cherished by everyone"* |
| 😲 | Surprise | *"Oh wow, I did NOT see that coming!"* |
| 😨 | Fear | *"I am terrified, my hands won't stop shaking"* |
| 😊 | Joy | *"Today was the best day of my life!"* |

---

## 📊 Dataset

- **Source:** Custom labeled text dataset (`train.txt`)
- **Format:** `text;emotion` (semicolon-separated)
- **Total Samples:** 16,000
- **Class Distribution:**

| Emotion | Samples |
|---------|---------|
| Joy | 5,362 |
| Sadness | 4,666 |
| Anger | 2,159 |
| Fear | 1,937 |
| Love | 1,304 |
| Surprise | 572 |

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF |
| Features | 13,359 |
| Tuning | GridSearchCV (C, solver) |
| Best Params | `C=10, solver=liblinear` |
| Test Accuracy | ~88% |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/johaibakhtar/EmoSense.git
cd EmoSense
```

### 2. Install dependencies
```bash
pip install streamlit scikit-learn joblib numpy
```

### 3. Run the app
```bash
streamlit run emotion_app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
EmoSense/
│
├── emotion_app.py                      # Streamlit UI app
├── NLP_Learnings.ipynb                 # Training notebook
├── EmotionClassifier_LogisticReg.joblib  # Trained model
├── tfidf_vectorizer.joblib             # Fitted TF-IDF vectorizer
├── train.txt                           # Training dataset
└── README.md
```

---

## 🖥️ App Features

- **Real-time prediction** — type any text and get instant emotion detection
- **Confidence scores** — probability breakdown across all 6 emotions
- **Sorted probability bars** — visual bar chart of all emotion scores
- **Example buttons** — 6 pre-loaded example sentences to try
- **Analysis history** — sidebar tracks your last 10 predictions
- **Dark UI** — clean, modern dark theme

---

## 🔮 Future Improvements

- [ ] **Deep Learning model** — replace Logistic Regression with BERT or DistilBERT for higher accuracy on ambiguous text
- [ ] **More emotion classes** — expand beyond 6 emotions (e.g. disgust, shame, guilt, excitement)
- [ ] **Multilingual support** — detect emotions in Urdu, Hindi, and other languages
- [ ] **Batch prediction** — upload a CSV file and classify multiple texts at once
- [ ] **Emotion timeline** — visualize how emotion changes across a long passage or conversation
- [ ] **REST API** — expose the model as a FastAPI endpoint for integration with other apps
- [ ] **Data balancing** — address class imbalance (Surprise has only 572 samples) using SMOTE or oversampling
- [ ] **Deployment** — host on Streamlit Cloud or HuggingFace Spaces for public access

---

## 👨‍💻 Author

**Johaib Akhtar**
- GitHub: [@johaibakhtar](https://github.com/johaibakhtar)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
