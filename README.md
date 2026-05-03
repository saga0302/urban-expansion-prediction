# 🌆 Satellite Imagery Analysis for Urban Expansion Prediction

Predicting urban growth patterns by 2030 using NASA satellite raster data and a U-Net convolutional neural network trained on non-Asia regions and evaluated on Asia.

---

## 📌 Project Overview

Urban expansion is one of the most significant drivers of environmental and infrastructural change globally. This project uses NASA's global urbanization probability raster dataset to train a deep learning model that predicts urban growth likelihood across the Asian continent.

The core idea: **train on the rest of the world, predict on Asia** — a true generalization test for the model.

---

## 🛰️ Data Source

- **Dataset:** [Global Grid of Probabilities of Urban Expansion to 2030](https://sedac.ciesin.columbia.edu/data/set/grump-v1-urban-extents) — NASA SEDAC
- **Format:** GeoTIFF (WGS84 projection)
- **Resolution:** Global raster with pixel-level urbanization probability values (0–255)

> The `.tif` file is not included in this repo due to size. Download it directly from NASA SEDAC and place it in the project root.

---

## 🧠 Model Architecture — U-Net

A U-Net architecture was chosen for its strength in spatial segmentation tasks, particularly where preserving fine-grained geographic structure matters.

```
Input (256×256×1)
    │
    ├── Encoder
    │     Conv2D(16) → Conv2D(16) → MaxPool
    │     Conv2D(32) → Conv2D(32) → MaxPool
    │     Conv2D(64) → Conv2D(64) → MaxPool
    │     Conv2D(128) + Dropout(0.3)
    │
    └── Decoder
          UpSample + Skip(64) → Conv2D(64) → Conv2D(64)
          UpSample + Skip(32) → Conv2D(32) → Conv2D(32)
          UpSample + Skip(16) → Conv2D(16) → Conv2D(16)
          Conv2D(1, sigmoid) → Output
```

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam (lr=1e-4)  
- **Tile Size:** 256×256 patches  
- **Early Stopping:** patience=3 on val_loss  

---

## 🔄 Pipeline

```
Raw GeoTIFF
    │
    ├── Crop Asia region (lat: 1–81, lon: 25–180)
    ├── Mask Asia from global data → Training set (non-Asia)
    │
    ├── Tile into 256×256 patches
    ├── Normalize (÷ 255)
    ├── Train/Val split (80/20)
    │
    ├── Train U-Net on non-Asia patches
    │
    └── Predict on Asia patches → Evaluate
```

---

## 📊 Results

| Metric | Score |
|---|---|
| **F1 Score** | **0.9846** |
| **Jaccard Index (IoU)** | **0.9696** |
| Final Val Accuracy | 97.59% |
| Final Val Loss | 0.0354 |

The model was trained for 20 epochs with batch size 32, achieving strong generalization from non-Asia training data to Asia predictions.

---

## 🗂️ Repository Structure

```
urban-expansion-prediction/
│
├── Urban_Expansion_Project_nasa_data.ipynb   # Full pipeline notebook
├── requirements.txt                           # Dependencies
├── .gitignore                                 # Excludes .tif and checkpoints
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/saga0302/urban-expansion-prediction.git
cd urban-expansion-prediction

# Install dependencies
pip install -r requirements.txt

# Download the NASA GeoTIFF and place it as:
# global-grid-prob-urban-expansion-2030-wgs84.tif

# Run the notebook
jupyter notebook Urban_Expansion_Project_nasa_data.ipynb
```

---

## 🧰 Tech Stack

`Python` `TensorFlow/Keras` `Rasterio` `NumPy` `Scikit-learn` `Matplotlib`

---

## 🔭 Future Work

- Incorporate multi-band satellite imagery (e.g., Landsat, Sentinel-2) for richer features
- Add temporal data to model year-over-year expansion trends
- Deploy as an interactive web map using Folium or Kepler.gl
- Extend predictions to 2050 using climate and population growth projections

---

## 👩‍💻 Author

**Sagarika Raju**  
MS Analytics, University of Southern California  
[GitHub](https://github.com/saga0302) · [LinkedIn](https://linkedin.com/in/sagarika-raju-ab28051a5/)
