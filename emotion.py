import streamlit as st
import joblib
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSense · Emotion Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Emotion metadata  (0=Sadness 1=Anger 2=Love 3=Surprise 4=Fear 5=Joy) ──────
EMOTIONS = {
    0: {"label": "Sadness",  "emoji": "😢", "color": "#5B8FDE"},
    1: {"label": "Anger",    "emoji": "😠", "color": "#E85D3A"},
    2: {"label": "Love",     "emoji": "❤️", "color": "#E8607A"},
    3: {"label": "Surprise", "emoji": "😲", "color": "#4ECFA8"},
    4: {"label": "Fear",     "emoji": "😨", "color": "#A67FDB"},
    5: {"label": "Joy",      "emoji": "😊", "color": "#F7C948"},
}

EXAMPLE_TEXTS = [
    "I can't stop crying, everything feels hopeless.",
    "I am absolutely furious, how dare they!",
    "I feel so loved and cherished by everyone.",
    "Oh wow, I did NOT see that coming at all!",
    "I am terrified, my hands won't stop shaking.",
    "Today was the best day of my life!",
]

# ── CSS (only styling, no content rendering) ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
    background-color: #07070f;
    color: #ddddf0;
}
.stApp { background: #07070f !important; }
.block-container { padding: 1.5rem 1.5rem 4rem !important; max-width: 780px !important; }

/* Header */
.emo-header { text-align: center; padding: 1.8rem 0 1.4rem; }
.emo-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.04em;
    background: linear-gradient(130deg, #ffffff 20%, #8080ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem;
    line-height: 1;
}
.emo-header .sub {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #44446a;
}

/* Textarea */
.stTextArea textarea {
    background: #0f0f1e !important;
    border: 1.5px solid #22223a !important;
    border-radius: 14px !important;
    color: #ddddf0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.92rem !important;
    padding: 1rem 1.1rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #6060ff !important;
    box-shadow: 0 0 0 3px rgba(96,96,255,0.12) !important;
}
.stTextArea label { display: none !important; }

/* Buttons */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    transition: transform 0.15s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

/* Metric boxes */
[data-testid="stMetric"] {
    background: #0f0f1e;
    border-radius: 14px;
    padding: 1.2rem !important;
    border: 1px solid #22223a;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #44446a !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

/* Progress bars */
.stProgress > div > div {
    border-radius: 999px !important;
    height: 8px !important;
}
.stProgress > div {
    border-radius: 999px !important;
    background: #1a1a2e !important;
    height: 8px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b0b18 !important;
    border-right: 1px solid #16162a !important;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Load model & vectorizer ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load("EmotionClassifier_LogisticReg.joblib")
        tfidf = joblib.load("tfidf_vectorizer.joblib")
    return model, tfidf


def predict(model, tfidf, text):
    X = tfidf.transform([text])
    pred_class = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return pred_class, proba


# ── Session state ──────────────────────────────────────────────────────────────
if "history"    not in st.session_state: st.session_state.history    = []
if "input_text" not in st.session_state: st.session_state.input_text = ""
if "result"     not in st.session_state: st.session_state.result     = None

# ── Sidebar history ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📜 Recent Analyses")
    if st.session_state.history:
        for item in st.session_state.history:
            st.markdown(f"**{item['emoji']} {item['emotion']}** — {item['confidence']:.0f}%")
            st.caption(item["text"])
            st.divider()
    else:
        st.caption("No analyses yet.")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="emo-header">
  <h1>EmoSense</h1>
  <div class="sub">Logistic Regression · 6 Emotions · ~88% Accuracy</div>
</div>
""", unsafe_allow_html=True)

# ── Load ───────────────────────────────────────────────────────────────────────
model, tfidf = load_artifacts()

# ── Input ──────────────────────────────────────────────────────────────────────
user_text = st.text_area(
    "Your text",
    value=st.session_state.input_text,
    placeholder="Type anything — a sentence, a tweet, a journal entry…",
    height=120,
    label_visibility="collapsed",
)

col1, col2 = st.columns([4, 1])
with col1:
    analyze = st.button("⚡  Analyze Emotion", use_container_width=True, type="primary")
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.session_state.input_text = ""
    st.session_state.result     = None
    st.session_state.history    = []
    st.rerun()

# ── Example buttons ────────────────────────────────────────────────────────────
st.caption("TRY AN EXAMPLE")
ex_cols = st.columns(3)
for i, ex in enumerate(EXAMPLE_TEXTS):
    with ex_cols[i % 3]:
        if st.button(ex[:26] + "…", key=f"ex_{i}", use_container_width=True):
            st.session_state.input_text = ex
            st.rerun()

# ── Run prediction ─────────────────────────────────────────────────────────────
if analyze:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        pred_class, proba = predict(model, tfidf, user_text)
        st.session_state.result = (pred_class, proba)
        meta = EMOTIONS[pred_class]
        st.session_state.history.insert(0, {
            "text":       user_text[:55] + ("…" if len(user_text) > 55 else ""),
            "emotion":    meta["label"],
            "emoji":      meta["emoji"],
            "color":      meta["color"],
            "confidence": proba[pred_class] * 100,
        })
        if len(st.session_state.history) > 10:
            st.session_state.history.pop()

# ── Result display (100% native Streamlit) ─────────────────────────────────────
if st.session_state.result:
    pred_class, proba = st.session_state.result
    meta       = EMOTIONS[pred_class]
    confidence = proba[pred_class] * 100

    st.divider()

    # Big result
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f"<p style='font-size:5rem; text-align:center; margin:0;'>{meta['emoji']}</p>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<p style='font-family:Syne,sans-serif; font-size:2.6rem; font-weight:800; "
            # f"color:{meta[\"color\"]}; margin:0; line-height:1;'>{meta['label']}</p>"
            f"<p style='color:#44446a; font-size:0.75rem; letter-spacing:0.12em; "
            f"text-transform:uppercase; margin-top:0.4rem;'>Confidence · {confidence:.1f}%</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Probability bars — all native
    st.caption("ALL EMOTION SCORES")
    for idx in np.argsort(proba)[::-1]:
        m    = EMOTIONS[idx]
        pct  = proba[idx] * 100
        bold = "**" if idx == pred_class else ""
        col_a, col_b, col_c = st.columns([2, 6, 1])
        with col_a:
            st.markdown(f"{m['emoji']} {bold}{m['label']}{bold}")
        with col_b:
            st.progress(float(proba[idx]))
        with col_c:
            st.markdown(f"`{pct:.1f}%`")

    st.divider()
    st.caption("EmoSense · LogisticRegression · TF-IDF · 13,359 features")
