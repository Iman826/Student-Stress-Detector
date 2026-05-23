# Multimodal Student Stress Detection System

This repository contains a **Multimodal Student Stress Detection System** that combines real-time facial emotion recognition (pictorial data) with an interactive survey (tabular data) to predict stress levels. It supports both a local command-line interface with a CV2 camera window and a full-featured modern web application interface.

## 🚀 Features

- **Real-Time Facial Expression Analysis:** Uses a ResNet-50 model trained on the RAF-DB dataset to detect faces and classify facial expressions at runtime (running on CPU or CUDA).
- **Survey-Based Stress Assessment:** Uses a PyTorch Neural Network trained on a student stress dataset to classify stress indicators from survey responses.
- **Weighted Fusion Engine:** Combines facial expression scores (25% weight) and tabular survey scores (75% weight) dynamically to output a robust fused stress score categorized from *Low* to *Severe*.
- **Interactive Web Interface:** A premium, modern web dashboard with live camera feed, real-time analytics graphs, responsive survey form, and clean status indicators.
- **CLI/GUI Application:** A quick-run Python script that displays a local CV2 window overlaying face rectangles, labels, and real-time stress levels.

---

## 📁 Project Structure

```
├── app.py                     # Flask web server & prediction API
├── fusion.py                  # Local CLI/GUI OpenCV interface
├── student_stress_model.pth   # Tabular neural network model weights (~10 KB)
├── best_rafdb_resnet50.pth    # ResNet-50 emotion recognition model weights (~94 MB) *
├── ema_weights.pth            # Alternative exponential moving average weights (~94 MB) *
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Files excluded from git
├── static/
│   └── index.html             # Single-page web app dashboard (HTML, CSS, JS)
└── Stress Indicators...csv    # Raw training dataset reference
```
*\* Note: Large model files are ignored by `.gitignore` by default to prevent repository bloat. See below on how to handle large files.*

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Set up a virtual environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: If you have a CUDA-enabled GPU and want to run predictions on hardware acceleration, install PyTorch with CUDA support matching your system configurations from [pytorch.org](https://pytorch.org/).*

### 4. Download Model Weights
Due to the file size limits of standard Git repositories (~100MB), the large PyTorch weights (`best_rafdb_resnet50.pth` and `ema_weights.pth`) are listed in `.gitignore`. 
- **Option A:** Download the weights from your external storage (e.g., Google Drive, HuggingFace, OneDrive) and place them directly in the root directory.
- **Option B (Git LFS):** If you wish to host them on GitHub, install [Git LFS](https://git-lfs.com/) and track them with:
  ```bash
  git lfs install
  git lfs track "*.pth"
  ```

---

## 💻 Running the App

### Option A: The Web Dashboard (Recommended)
Launch the Flask web server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```
- Click **Start Camera** to begin the live emotion analyzer.
- Fill out the survey form.
- Click **Calculate Fused Stress Score** to view the combined analytics and diagnostic charts.

### Option B: Local CLI & OpenCV Window
Run the standalone fusion script:
```bash
python fusion.py
```
- A window will pop up showing your webcam feed with face bounding boxes and current emotion labels.
- Go to the terminal/command line and answer the survey questions.
- Once completed, the camera window will overlay your real-time fused stress score.
- Press **Q** on the camera window to exit.

---

## 📤 Uploading to GitHub

To upload this repository to your GitHub profile, run the following commands in the project directory:

```bash
# Initialize git repository
git init

# Add all files to staging (large model weights will be ignored automatically by .gitignore)
git add .

# Create the initial commit
git commit -m "Initial commit: Multimodal student stress detection system"

# Create a repository on GitHub, then link it
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push the code to GitHub
git push -u origin main
```
