# ScoliSnap

A Python-based computer vision tool that detects spinal curvature (Scoliosis) from 2D back-view images. Unlike deep learning approaches that require massive labeled datasets, this project uses **classical image processing** and **geometric analysis** to quantify asymmetry.

## 🚀 How to Run Locally

**1. Setup Virtual Environment** (if you haven't already)**:**

```bash
# Windows
#

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Application**

```bash
streamlit run main.py
```