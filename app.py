# ============================================================
# app.py — Emotion Recognition App — Premium UI
# ============================================================
import streamlit as st
import numpy as np
import pickle
import re
import string
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EmoSense AI",
    page_icon="🎭",
    layout="centered"
)

# ============================================================
# FULL CUSTOM CSS — Premium Dark UI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, .stApp {
    background: #080b14 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e8eaf0 !important;
}

/* ── Hide Streamlit chrome ───────────────────── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container {
    padding: 2rem 1.5rem 4rem !important;
    max-width: 780px !important;
}

/* ── Hero Header ─────────────────────────────── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.35);
    color: #a78bfa;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 100px;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 6vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.hero-sub {
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── Emotion Pills Row ───────────────────────── */
.pills-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin: 1.8rem 0 2.5rem;
}
.pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.82rem;
    color: #9ca3af;
    letter-spacing: 0.02em;
}

/* ── Input Card ──────────────────────────────── */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.input-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #6b7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Streamlit textarea override ─────────────── */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: #f1f5f9 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    resize: none !important;
    transition: border-color 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: rgba(139,92,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #374151 !important; }
.stTextArea label { display: none !important; }

/* ── Analyze Button ──────────────────────────── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 32px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124,58,237,0.55) !important;
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result Hero Card ────────────────────────── */
.result-hero {
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 2rem 0 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 24px;
    padding: 1px;
    background: linear-gradient(135deg, var(--ec), transparent, var(--ec));
    -webkit-mask: linear-gradient(#fff 0 0) content-box,
                  linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0.6;
}
.result-emoji {
    font-size: 5rem;
    line-height: 1;
    margin-bottom: 1rem;
    display: block;
    filter: drop-shadow(0 0 30px var(--ec));
    animation: pulse-emoji 2s ease-in-out infinite;
}
@keyframes pulse-emoji {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.08); }
}
.result-emotion {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--ec);
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.6rem;
    text-shadow: 0 0 40px var(--ec-dim);
}
.result-conf {
    font-size: 0.95rem;
    color: #6b7280;
    font-weight: 400;
}
.result-conf span {
    color: var(--ec);
    font-weight: 700;
}

/* ── Scores Section ──────────────────────────── */
.scores-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    color: #4b5563;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 2rem 0 1.2rem;
}
.score-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
    padding: 14px 18px;
    background: rgba(255,255,255,0.025);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
    transition: background 0.2s;
}
.score-row.top-row {
    background: rgba(255,255,255,0.055);
    border-color: rgba(255,255,255,0.1);
}
.score-icon { font-size: 1.4rem; flex-shrink: 0; }
.score-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #d1d5db;
    width: 90px;
    flex-shrink: 0;
}
.score-label.top-label { color: #f9fafb; }
.score-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: var(--bar-color);
    box-shadow: 0 0 10px var(--bar-color);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.score-pct {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--bar-color);
    width: 46px;
    text-align: right;
    flex-shrink: 0;
}
.top-badge {
    background: var(--bar-color);
    color: #000;
    font-size: 0.6rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 100px;
    flex-shrink: 0;
}

/* ── History ─────────────────────────────────── */
.hist-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    color: #4b5563;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 2.5rem 0 1rem;
}
.hist-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 18px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    margin-bottom: 8px;
}
.hist-emoji { font-size: 1.3rem; }
.hist-info { flex: 1; min-width: 0; }
.hist-emotion {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #e5e7eb;
}
.hist-text {
    font-size: 0.78rem;
    color: #4b5563;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 2px;
}
.hist-pct {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    color: #6b7280;
    flex-shrink: 0;
}

/* ── Clear button ────────────────────────────── */
.stButton.clear-btn > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #6b7280 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    box-shadow: none !important;
    letter-spacing: 0.02em !important;
}
.stButton.clear-btn > button:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #9ca3af !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Divider ─────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 0.5rem 0 !important; }

/* ── Warning / Error ─────────────────────────── */
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
MAX_SEQ_LEN = 60

EMOTION_MAP = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

EMOTION_DATA = {
    'Sadness':  {'emoji': '😢', 'color': '#60a5fa', 'dim': 'rgba(96,165,250,0.2)'},
    'Joy':      {'emoji': '😄', 'color': '#fbbf24', 'dim': 'rgba(251,191,36,0.2)'},
    'Love':     {'emoji': '❤️',  'color': '#f87171', 'dim': 'rgba(248,113,113,0.2)'},
    'Anger':    {'emoji': '😠', 'color': '#fb923c', 'dim': 'rgba(251,146,60,0.2)'},
    'Fear':     {'emoji': '😨', 'color': '#c084fc', 'dim': 'rgba(192,132,252,0.2)'},
    'Surprise': {'emoji': '😲', 'color': '#34d399', 'dim': 'rgba(52,211,153,0.2)'},
}

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        model = load_model('lstm_emotion_model.keras')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

with st.spinner("Warming up EmoSense..."):
    model, tokenizer, error = load_artifacts()

if error:
    st.error(f"Could not load model: {error}")
    st.info("Make sure `lstm_emotion_model.keras` and `tokenizer.pkl` are in the same folder.")
    st.stop()

# ============================================================
# PREPROCESSING
# ============================================================
STOP_WORDS = set(stopwords.words('english'))
KEEP_WORDS = {
    'not','no','never','nor','neither','nothing','nobody',
    'nowhere','hardly','barely','scarcely','very','so','too',
    'quite','really','absolutely','extremely','totally',
    'completely','deeply','truly','feel','feeling','felt',
    'feels','happy','sad','angry','scared','afraid','love',
    'hate','fear','joy','hurt','pain','glad','upset','worried',
    'anxious','excited','surprised','shocked','furious',
    'terrified','miserable','wonderful','awful'
}
CONTRACTIONS = {
    "i'm":"i am","i've":"i have","i'd":"i would","i'll":"i will",
    "you're":"you are","it's":"it is","don't":"do not",
    "doesn't":"does not","didn't":"did not","can't":"cannot",
    "won't":"will not","wouldn't":"would not","shouldn't":"should not",
    "couldn't":"could not","isn't":"is not","aren't":"are not",
    "wasn't":"was not","weren't":"were not","hasn't":"has not",
    "hadn't":"had not","haven't":"have not","that's":"that is",
    "there's":"there is","he's":"he is","she's":"she is",
    "we're":"we are","they're":"they are","we've":"we have",
    "they've":"they have","you've":"you have","we'll":"we will",
    "they'll":"they will","i'm":"i am"
}

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    for c, e in CONTRACTIONS.items():
        text = text.replace(c, e)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS or w in KEEP_WORDS]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

def predict_emotion(text):
    cleaned  = clean_text(text)
    seq      = tokenizer.texts_to_sequences([cleaned])
    seq      = pad_sequences(seq, maxlen=MAX_SEQ_LEN,
                             padding='post', truncating='post')
    proba    = model.predict(seq, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    return EMOTION_MAP[pred_idx], proba

# ============================================================
# SESSION STATE
# ============================================================
if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ Deep Learning NLP</div>
    <div class="hero-title">EmoSense AI</div>
    <div class="hero-sub">Understand the emotion behind every word</div>
</div>

<div class="pills-row">
    <span class="pill">😢 Sadness</span>
    <span class="pill">😄 Joy</span>
    <span class="pill">❤️ Love</span>
    <span class="pill">😠 Anger</span>
    <span class="pill">😨 Fear</span>
    <span class="pill">😲 Surprise</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="input-label">Your Message</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="input",
    placeholder="Write anything... e.g. I can't believe this happened, I'm completely shocked!",
    height=130,
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    btn = st.button("✦  Analyze Emotion", use_container_width=True)

# ============================================================
# PREDICTION OUTPUT
# ============================================================
if btn:
    if not user_input.strip():
        st.warning("Please write something first.")
    else:
        with st.spinner("Reading between the lines..."):
            emotion, proba = predict_emotion(user_input)

        ed   = EMOTION_DATA[emotion]
        conf = float(np.max(proba)) * 100

        # ── Result Hero Card ─────────────────────────────────
        st.markdown(f"""
        <div class="result-hero" style="
            background: radial-gradient(ellipse at center, {ed['dim']} 0%, rgba(8,11,20,0) 70%);
            --ec: {ed['color']};
            --ec-dim: {ed['dim']};
        ">
            <span class="result-emoji">{ed['emoji']}</span>
            <div class="result-emotion">{emotion}</div>
            <div class="result-conf">
                Model confidence &nbsp;
                <span>{conf:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Score Bars ───────────────────────────────────────
        st.markdown('<div class="scores-title">All Emotion Scores</div>',
                    unsafe_allow_html=True)

        sorted_idx = np.argsort(proba)[::-1]
        for rank, idx in enumerate(sorted_idx):
            emo       = EMOTION_MAP[idx]
            conf_val  = float(proba[idx])
            d         = EMOTION_DATA[emo]
            is_top    = rank == 0
            top_cls   = "top-row" if is_top else ""
            lbl_cls   = "top-label" if is_top else ""
            badge_html = f'<span class="top-badge" style="--bar-color:{d["color"]}">TOP</span>' if is_top else ""

            st.markdown(f"""
            <div class="score-row {top_cls}">
                <span class="score-icon">{d['emoji']}</span>
                <span class="score-label {lbl_cls}">{emo}</span>
                <div class="score-bar-bg">
                    <div class="score-bar-fill"
                         style="width:{conf_val*100:.1f}%;
                                --bar-color:{d['color']};
                                background:{d['color']};
                                box-shadow: 0 0 8px {d['color']}88;">
                    </div>
                </div>
                <span class="score-pct" style="--bar-color:{d['color']};">
                    {conf_val*100:.1f}%
                </span>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

        # ── Save History ─────────────────────────────────────
        st.session_state.history.insert(0, {
            'text':       user_input[:60] + ('...' if len(user_input) > 60 else ''),
            'emotion':    emotion,
            'emoji':      ed['emoji'],
            'confidence': conf
        })
        if len(st.session_state.history) > 6:
            st.session_state.history.pop()

# ============================================================
# HISTORY
# ============================================================
if st.session_state.history:
    st.markdown('<div class="hist-title">Recent Predictions</div>',
                unsafe_allow_html=True)

    for item in st.session_state.history:
        st.markdown(f"""
        <div class="hist-row">
            <span class="hist-emoji">{item['emoji']}</span>
            <div class="hist-info">
                <div class="hist-emotion">{item['emotion']}</div>
                <div class="hist-text">{item['text']}</div>
            </div>
            <span class="hist-pct">{item['confidence']:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, cc, _ = st.columns([2, 1, 2])
    with cc:
        if st.button("Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()
