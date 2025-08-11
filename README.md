<h1 align="center">FLUENCIS</h1>
<p align="center">Where speech meets precision</p>
# Stuttering Detection & Correction

A machine learning and deep learning-based system for detecting and classifying stuttering types such as **Prolongation**, **Block**, **Word Repetition**, **Sound Repetition**, and **Interjection** from speech audio files.  
This project uses the **SEP-28k dataset**, multiple feature extraction methods (MFCC, ZCR, Jitter, Shimmer), and models like **Decision Tree**, **Random Forest**, **SVM**, and **CNN** to achieve high detection accuracy.

---

## Features
- **Multi-type Stuttering Detection**: Detects Prolongation, Block, Word Repetition, Sound Repetition, Interjection.
- **Multiple Models**: ML (SVM, Decision Tree, Random Forest, KNN) & DL (CNN).
- **Feature Extraction**: MFCC, Zero Crossing Rate (ZCR), Jitter, Shimmer.
- **Data Balancing**: Uses SMOTE to handle class imbalance.
- **Web Interface**: Flask app for real-time predictions from uploaded audio.
- **High Accuracy**: Achieves up to 90%+ accuracy for certain stuttering types.

---

## Dataset
**SEP-28k Dataset**
- Over 28,000 annotated audio clips (.wav format) from natural conversations.
- Labels: Prolongation, Block, Word Repetition, Sound Repetition, Interjection.
- Quality indicators: `PoorAudioQuality`, `DifficultToUnderstand`.
- Preprocessing: Noise reduction, normalization, segmentation.

---

## Methodology
### 1. Data Cleaning
- Removed invalid audio files (44 bytes – no audio data).
- Excluded low-quality or irrelevant samples.

### 2. Feature Extraction
- **MFCC**: Captures timbral and spectral characteristics.
- **ZCR**: Measures sign changes in waveform.
- **Jitter & Shimmer**: Pitch and amplitude variations.
- **Log-Mel Spectrogram** (for CNN input).

### 3. Label Transformation
- Binary conversion for each stuttering type.
- Combined `Stutter` label for overall detection.

### 4. Data Balancing
- Applied **SMOTE** for synthetic oversampling.

### 5. Model Training
- **ML Models**: SVM (RBF kernel), Decision Tree, Random Forest.
- **DL Model**: CNN for spectrogram classification.

### 6. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score.
- Confusion matrices for detailed error analysis.

---

## Model Performance
| Stuttering Type | Best Model       | Accuracy |
|-----------------|-----------------|----------|
| Prolongation    | Random Forest   | 78%      |
| Block           | Random Forest   | 65%      |
| Word Repetition | Random Forest   | 90%      |
| Sound Repetition| Random Forest   | 87%      |
| Overall         | Ensemble Model  | 87%      |
| CNN (Prolong.)  | CNN Model       | 72%      |

---

## Technologies Used
- **Languages**: Python, HTML, CSS
- **Frameworks**: Flask, scikit-learn, TensorFlow/Keras
- **Libraries**: NumPy, Pandas, Librosa, Matplotlib, SMOTE
- **Dataset**: SEP-28k

---

## Installation
bash
# Clone repository
git clone https://github.com/<your-username>/<repo-name>.git

# Navigate to folder
cd <repo-name>

# Install dependencies
pip install -r requirements.txt
`

---

## Usage

1. **Prepare Dataset**

   * Place SEP-28k audio files in `data/audio/`
   * Place labels CSV in `data/labels/`

2. **Train Models**

   bash
   python train_ml_models.py
   python train_cnn_model.py
   

3. **Run Flask App**

   bash
   python app.py
   

   Access web interface at: `http://127.0.0.1:5000`

4. **Upload Audio File**

   * System predicts stuttering type(s) and displays results.

---

## Results

* **Overall Accuracy**: \~87%
* **Best Detection**: Word Repetition (90% - Random Forest)
* **DL Strength**: CNN excels in detecting subtle patterns like Prolongations.

---

## Future Scope

* Implement **Transformer-based models** for higher accuracy.
* Add **real-time speech streaming detection**.
* Develop a **stuttering correction module**.
* Expand to **multi-language support**.

---

## Contributors

* **Anukta Roy**
* **Shreya Saha**
* **Disha Makal**
* **Swarup Saw**
* **Rahul Pal**

**Mentor**: Prof. Dr. Sudipta Bhattacharya, Dept. of CSE & IT, Bengal Institute of Technology, Kolkata.

---
<h3>Preview:</h3>


https://github.com/user-attachments/assets/5644e8ca-fd8e-4c23-a22c-53f627569903

