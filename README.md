# 🎭 EmoSense AI — Emotion Recognition from Text using RNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=for-the-badge&logo=streamlit)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?style=for-the-badge)
<!---![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)--->

**A Deep Learning project that detects human emotions from text using Bidirectional LSTM and GRU neural networks.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Emotions Detected](#-emotions-detected)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Model Architecture](#-model-architecture)
- [Text Preprocessing](#-text-preprocessing)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [Streamlit Web App](#-streamlit-web-app)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Future Improvements](#-future-improvements)

---

## 🧠 Overview

**EmoSense AI** is an end-to-end Natural Language Processing (NLP) project that identifies the underlying emotional state in a given sentence or paragraph. Unlike basic sentiment analysis (positive/negative), this project classifies text into **6 distinct emotion categories** using a deep learning sequence model.

The project includes:
- Full data preprocessing pipeline
- Bidirectional LSTM + GRU model training in Google Colab
- Interactive web application built with Streamlit
- Real-time emotion prediction with confidence scores

---

## 🎯 Emotions Detected

| Label | Emotion | Emoji |
|-------|---------|-------|
| 0 | Sadness | 😢 |
| 1 | Joy | 😄 |
| 2 | Love | ❤️ |
| 3 | Anger | 😠 |
| 4 | Fear | 😨 |
| 5 | Surprise | 😲 |

---

## 📁 Project Structure

```
EMOTION-DETECTION/
│
├── 📂 data set/                          # Raw dataset files
│   ├── training.csv                      # 16,000 training samples
│   ├── validation.csv                    # 2,000 validation samples
│   └── test.csv                          # 2,000 test samples
│
├── 📂 myenv/                             # Python virtual environment
│
├── 🐍 app.py                             # Streamlit web application
├── 📓 Emotion_Recognition_RNN.ipynb      # Google Colab training notebook
│
├── 🤖 new_lstm_emotion_model.keras        # Best trained LSTM model (v2)
├── 🤖 lstm_emotion_model.keras            # Initial LSTM model (v1)
├── 🤖 gru_emotion_model.keras             # GRU model
│
├── 🔤 new_tokenizer.pkl                   # Tokenizer for new model (v2)
├── 🔤 tokenizer.pkl                       # Tokenizer for initial model (v1)
│
├── ⚙️ emotion_config.json                 # Model configuration file
├── 📋 requirements.txt                    # Python dependencies
├── 📝 sample.txt                          # Sample test sentences
└── 📖 README.md                           # Project documentation
```

> **Note:** The project currently uses `new_lstm_emotion_model.keras` and `new_tokenizer.pkl` for predictions as these are the improved retrained models with better accuracy.

---

## 📊 Dataset

| Split | Samples |
|-------|---------|
| Training | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |
| **Total** | **20,000** |

**Class Distribution in Training Set:**

| Emotion | Count |
|---------|-------|
| Joy | 5,362 |
| Sadness | 4,666 |
| Anger | 2,159 |
| Fear | 1,937 |
| Love | 1,304 |
| Surprise | 572 |

Each sample contains two columns — `text` (the sentence) and `label` (the emotion class 0–5).

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10** | Core programming language |
| **TensorFlow / Keras** | Deep learning model building & training |
| **NLTK** | Text preprocessing, stopwords, lemmatization |
| **Scikit-learn** | Evaluation metrics, class weights |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Data visualization |
| **WordCloud** | Emotion word cloud visualization |
| **Streamlit** | Interactive web application |
| **Google Colab** | Model training environment |

---

## 🏗️ Model Architecture

Two models were built and compared:

### Model 1 — Bidirectional LSTM
```
Input → Embedding(20000, 128) → SpatialDropout1D(0.3)
     → BiLSTM(128, return_seq=True) → BiLSTM(64)
     → Dropout(0.3) → Dense(64, relu)
     → Dropout(0.15) → Dense(6, softmax)
```

### Model 2 — Best Model (BiLSTM + BiGRU Combined) ✅
```
Input → Embedding(20000, 128) → SpatialDropout1D(0.3)
     → BiLSTM(128, return_seq=True) ──┐
     → BiGRU(128, return_seq=True)  ──┤
     → GlobalAvgPool + GlobalMaxPool  ─┤ (all 4 pooled)
     → Concatenate → Dense(256, relu) ─┘
     → BatchNorm → Dropout(0.4)
     → Dense(128, relu) → BatchNorm → Dropout(0.3)
     → Dense(64, relu) → Dropout(0.2)
     → Dense(6, softmax)
```

**Model Configuration:**

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 20,000 |
| Embedding Dimension | 128 |
| Max Sequence Length | 60 |
| Loss Function | Categorical Cross-Entropy |
| Optimizer | Adam (lr = 5e-4) |
| Batch Size | 32 |
| Max Epochs | 30 |
| Activation (Output) | Softmax |

**Callbacks Used:**
- `EarlyStopping` — patience 5, monitors val_accuracy
- `ModelCheckpoint` — saves best model only
- `ReduceLROnPlateau` — halves LR when val_loss stalls

---

## 🔄 Text Preprocessing

The following steps are applied to every input sentence before prediction:

```
Raw Text
   ↓  Lowercase
   ↓  Expand Contractions  (don't → do not, i'm → i am)
   ↓  Remove URLs
   ↓  Remove Non-ASCII Characters
   ↓  Remove HTML Tags
   ↓  Remove Numbers
   ↓  Remove Punctuation
   ↓  Remove Stopwords (keeping negation + emotion words)
   ↓  Lemmatization  (running → run, terrified → terrify)
   ↓  Tokenization + Sequence Padding (maxlen=60)
Clean Tokens → Model Input
```

**Words always kept (not removed as stopwords):**
- Negations: `not, no, never, nor, neither`
- Intensifiers: `very, so, really, absolutely, extremely`
- Emotion words: `happy, sad, angry, scared, love, fear, hurt`

---

## 🚀 Model Training

Training was done in **Google Colab** using the notebook `Emotion_Recognition_RNN.ipynb`.

**Key training decisions:**
- **Class weights** were applied to fix the Joy-bias caused by class imbalance
- **Smaller batch size (32)** was used for better learning on rare emotions
- **Bidirectional layers** capture both forward and backward context
- **Parallel LSTM + GRU branches** with pooling for richer feature extraction

---

## 📈 Model Evaluation

Evaluation metrics used:

- ✅ Accuracy
- ✅ Loss Curves (Training vs Validation)
- ✅ Classification Report (Precision, Recall, F1 per class)
- ✅ Confusion Matrix (raw counts + percentage)
- ✅ Per-class F1 Score comparison (LSTM vs GRU)
- ✅ Manual testing with custom sentences

---

## 🌐 Streamlit Web App

The `app.py` file provides a fully interactive web interface:

**Features:**
- 🎨 Premium dark UI with custom CSS and Google Fonts
- ⚡ Real-time emotion prediction
- 📊 Confidence score bars for all 6 emotions
- 🕘 Last 6 prediction history
- 🔍 Text preprocessing visualization
- 📱 Responsive centered layout

**App loads:**
- `new_lstm_emotion_model.keras` — improved LSTM model
- `new_tokenizer.pkl` — matching tokenizer

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

### 2. Create Virtual Environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# Mac / Linux
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

Open your browser at → `http://localhost:8501`

---

## 📋 Requirements

```
streamlit==1.32.0
tensorflow==2.15.0
numpy==1.26.4
nltk==3.8.1
scikit-learn==1.4.0
pandas
matplotlib
seaborn
wordcloud
```

---

## 🧪 Sample Test Sentences

| Sentence | Expected Emotion |
|----------|-----------------|
| I am so happy today, everything is going great! | 😄 Joy |
| I feel so sad and lonely, no one understands me. | 😢 Sadness |
| This makes me absolutely furious, how dare they! | 😠 Anger |
| I'm terrified about what might happen next. | 😨 Fear |
| Oh wow, I never expected that to happen at all! | 😲 Surprise |
| I love spending time with my family, it's wonderful. | ❤️ Love |

---

## 🔮 Future Improvements

- [ ] Add multi-label emotion classification support
- [ ] Integrate GloVe / Word2Vec pre-trained embeddings
- [ ] Add Transformer-based model (BERT) for comparison
- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces
- [ ] Add support for multiple languages
- [ ] Batch prediction from uploaded CSV files
- [ ] Add explainability (attention visualization)

---

## 👨‍💻 Author

**PAVAN AHIRE**
- GitHub: [@your-username](https://github.com/pavan-ahire)
- LinkedIn: [your-linkedin](https://www.linkedin.com/in/pavan-ahire-260940364/)

---
<!---
## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify and distribute.
--->
---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

Made with ❤️ using TensorFlow + Streamlit

</div>
