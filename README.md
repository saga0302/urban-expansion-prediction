# 🌆 Urban Expansion Prediction Using Multi-Source Satellite Data
### Predicting Urban Growth Across Asia by 2030 using Deep Learning

**Author:** Sagarika Raju | MS Analytics, University of Southern California

---

## 📌 Project Overview

Urban expansion is one of the most significant drivers of environmental and infrastructural change globally. This project builds a **U-Net deep learning model** that predicts urban growth probability across Asia by 2030 using 4 satellite-derived feature layers as input.

**Key design principle:** The model takes real-world geospatial features (population density, nighttime lights, elevation, vegetation) as input and learns to predict NASA's urban expansion probability — making it a genuine predictive pipeline, not a reconstruction exercise.

---

## 🛰️ Data Sources

| Layer | Dataset | Agency | Use |
|---|---|---|---|
| Urban Expansion 2030 | Global Grid of Probabilities | NASA SEDAC | Target variable |
| Population Density | GPW v4 2020 | NASA SEDAC | Input feature |
| Nighttime Lights | VIIRS DNB 2020 | NOAA | Input feature |
| Elevation | SRTM 1 Arc-Second | NASA/USGS | Input feature |
| NDVI Vegetation | MOD13A3 2020 | NASA MODIS | Input feature |

> All feature layers downloaded via Google Earth Engine API and clipped to Asia (lat: 1–81, lon: 25–180)

---

## 🔍 Key EDA Findings

Before modeling, EDA revealed critical insights that directly shaped modeling decisions:

| Finding | Value | Decision |
|---|---|---|
| Class imbalance | 91:1 non-urban to urban | Weighted loss (25x penalty) |
| Population correlation | r=0.44, 22.85x higher in cities | Strongest predictor |
| Nighttime lights ratio | 8.95x higher in urban areas | Log normalization |
| Elevation in cities | 0.39x (cities on flat land) | Keep as inverse feature |
| All features right-skewed | High std, low median | Log normalize pop & lights |

---

## 🧠 Model Architecture — U-Net

```
Input (64×64×4)  ← 4 satellite feature bands
    │
    ├── Encoder
    │     Conv2D(32) + BatchNorm → MaxPool
    │     Conv2D(64) + BatchNorm → MaxPool
    │     Conv2D(128) + BatchNorm → MaxPool
    │     Conv2D(256) + Dropout(0.3)  ← Bottleneck
    │
    └── Decoder
          UpSample + Skip(128) → Conv2D(128)
          UpSample + Skip(64)  → Conv2D(64)
          UpSample + Skip(32)  → Conv2D(32)
          Conv2D(1, sigmoid)   → Output probability map
```

**Total parameters:** 1,948,065

---

## 🔄 Pipeline

```
Raw GeoTIFF Files (5 layers)
    │
    ├── EDA → Discover class imbalance (91:1), feature correlations
    │
    ├── Clean → Remove nodata values, clip to valid ranges
    │
    ├── Normalize → Log scale (pop, lights) | Min-max (elev, ndvi)
    │
    ├── Patch → 64×64 tiles with 50% overlap + urban oversampling (5x)
    │           Total: 10,705 patches | Train: 8,564 | Val: 2,141
    │
    ├── Train U-Net → Weighted BCE loss (25x urban penalty)
    │                 30 epochs | Batch size 32 | Adam lr=1e-4
    │
    └── Evaluate → F1, IoU, Precision, Recall at optimal threshold
```

---

## 📊 Results

| Metric | Score |
|---|---|
| **F1 Score** | **0.7680** |
| **Jaccard Index (IoU)** | **0.6233** |
| **Recall** | **0.9621** |
| **Precision** | **0.6390** |
| Val Accuracy | 91.38% |
| Best Val Loss | 0.1699 |
| Total Epochs | 30 |

**Threshold:** 0.5 (optimized via threshold sweep from 0.05–0.5)

### Why these metrics matter:
- **Recall of 96.2%** — the model correctly identifies 96% of areas that will urbanize by 2030
- **IoU of 0.623** — strong spatial overlap between predicted and actual urban zones
- **F1 of 0.768** — robust balance between precision and recall despite 91:1 class imbalance

---

## 🗂️ Repository Structure

```
urban-expansion-prediction/
│
├── data/
│   ├── raw/                  ← downloaded .tif files (not in git)
│   └── processed/
│
├── notebooks/
│   └── urban_expansion_unet.ipynb   ← full pipeline notebook
│
├── models/                   ← saved model weights (not in git)
│
├── outputs/
│   └── figures/
│       ├── 01_cleaned_layers.png
│       ├── 02_normalized_layers.png
│       └── 03_predictions.png
│
├── src/
│   └── download_data.py      ← Earth Engine download script
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/saga0302/urban-expansion-prediction.git
cd urban-expansion-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download satellite data (requires Google Earth Engine account)
python src/download_data.py

# Run the notebook
jupyter notebook notebooks/urban_expansion_unet.ipynb
```

---

## 🧰 Tech Stack

`Python` `TensorFlow/Keras` `Rasterio` `Google Earth Engine` `NumPy` `Scikit-learn` `Scikit-image` `Matplotlib`

---

## 🔭 Future Work

- Incorporate temporal data (2000→2010→2020) for true time-series prediction
- Add road network proximity as additional feature
- Deploy as interactive web map using Folium or Kepler.gl
- Extend to global scale with cloud-based training (Google Colab Pro)
- Experiment with attention-based U-Net for better urban boundary detection

---

## 👩‍💻 Author

**Sagarika Raju**
MS Analytics, University of Southern California
[GitHub](https://github.com/saga0302) · [LinkedIn](https://linkedin.com/in/sagarika-raju-ab28051a5/)
