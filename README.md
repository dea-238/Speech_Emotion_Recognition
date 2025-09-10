# Speech Emotion Recognition

An advanced **Speech Emotion Recognition (SER)** model using a novel approach. It fuses standard acoustic features with phonetic metrics (Jitter, Shimmer, HNR) to capture subtle vocal cues. A custom 1D CNN with a self-attention mechanism processes these rich features, intelligently focusing on the most salient emotional moments in speech.

---

## 📌 Enhanced Speech Emotion Recognition with Attention-Based CNN and Phonetic Features

### 1. Project Overview

This project presents a novel and robust approach to **Speech Emotion Recognition (SER)**. It goes beyond standard methodologies by integrating advanced, phonetically-grounded voice features with a sophisticated deep learning architecture. The model is trained on a comprehensive, aggregated corpus from four diverse public datasets to accurately classify seven distinct emotional states from raw audio files.

The core of this project is the `speech-emotion-recognition.ipynb` notebook, which provides a complete, end-to-end pipeline from data preparation to model training, evaluation, and saving.

---

### 2. The Novelty: What Makes This Project Unique?

#### 🎤 Advanced Phonetic Feature Engineering

Instead of relying solely on common acoustic features like MFCCs, this model enriches the input data with voice quality metrics typically used in clinical phonetic analysis. These are extracted using the `praat-parselmouth` library, the gold standard for phonetic analysis.

- **Jitter (Voice Shakiness):** Measures micro-variations in vocal pitch, often linked to emotions like fear or sadness.
- **Shimmer (Volume Wavering):** Captures tiny fluctuations in vocal amplitude, indicative of emotional instability.
- **Harmonics-to-Noise Ratio (HNR):** Quantifies the clarity versus breathiness of the voice, distinguishing clear, angry tones from breathy, sad ones.

#### 🧠 Intelligent CNN with Self-Attention Mechanism

The model architecture is a custom-built **1D Convolutional Neural Network (CNN)** enhanced with a **Self-Attention** layer.

- **CNN Layers:** Act as powerful feature extractors, identifying complex local patterns within the rich feature set.
- **Self-Attention Layer:** The model dynamically assigns importance scores to different parts of the audio sequence. It automatically focuses on the most emotionally salient moments in an utterance, similar to how humans interpret emotional cues, leading to more accurate and context-aware predictions.

---

### 3. Methodology Pipeline

The project follows a systematic workflow:

1. **Data Aggregation:** Combines four renowned datasets to create a large and diverse corpus:
   - **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
   - **TESS** (Toronto Emotional Speech Set)
   - **SAVEE** (Surrey Audio-Visual Expressed Emotion)

2. **Preprocessing & Augmentation:** Audio files are cleaned and expanded using techniques like:
   - Noise injection
   - Pitch shifting  
   These augmentations help improve model generalization.

3. **Hybrid Feature Extraction:** Phonetic features (Jitter, Shimmer, HNR) are extracted and concatenated with standard features (MFCC, ZCR, RMSE) to create a comprehensive feature vector.

4. **Model Training:** The CNN with Self-Attention model is trained on the prepared data using callbacks like early stopping and learning rate reduction to optimize performance.

5. **Evaluation:** The model's accuracy is analyzed using confusion matrices and classification reports, assessing its performance across emotion categories.

6. **Saving Artifacts:** The trained model and supporting objects are saved for easy deployment and inference.

---

### 4. How to Use This Project

#### ✅ Prerequisites

You need Python 3 and the following libraries. Install them using:

```bash
pip install tensorflow keras librosa pandas scikit-learn seaborn matplotlib praat-parselmouth
