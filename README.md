<div align="center">

# 🛰️ EuroSAT Land-Use Classifier

### Teaching a CNN to read the Earth from space, one 64×64 tile at a time.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Ready-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Classes](https://img.shields.io/badge/Classes-10-8A2BE2?style=for-the-badge)
![Images](https://img.shields.io/badge/Images-27%2C000-2ea44f?style=for-the-badge)

<br>

**Sentinel-2 satellite imagery → a compact, regularised CNN → 10 land-use classes.**
No pretrained weights. No giant model. Just a clean pipeline you can read in one sitting.

</div>

---

## ✨ Highlights

| | |
|---|---|
| 🧠 **Lightweight CNN** | ~306K parameters (~1.2 MB). Three double-conv blocks with BatchNorm and Global Average Pooling. |
| 🔄 **Orientation-safe augmentation** | Flips, rotation and zoom, which suit satellite tiles that have no "up". |
| 🎯 **Stratified 70 / 15 / 15 split** | Every class is evenly represented in train, validation and test. The test set is only touched once, at the end. |
| 🏷️ **Labels from filenames** | The dataset is flat (27,000 JPGs in one folder), so classes are parsed from names like `Forest_123.jpg`. |
| 🛡️ **Fail-fast sanity checks** | Class counts are printed and the run stops if label parsing looks wrong, so there are no silent one-class bugs. |
| 📊 **Proper evaluation** | Per-class precision/recall/F1 and a confusion matrix, not just a single accuracy number. |
| ⏱️ **Smart training** | Early stopping, learning-rate decay on plateau, and best-checkpoint saving. |

---

## 🗺️ The 10 Classes

`AnnualCrop` · `Forest` · `HerbaceousVegetation` · `Highway` · `Industrial` · `Pasture` · `PermanentCrop` · `Residential` · `River` · `SeaLake`

---

## 🏗️ Architecture

```text
Input (64×64×3)
   │
   ├─ Augmentation ── RandomFlip · RandomRotation · RandomZoom
   ├─ Rescaling (1/255)
   │
   ├─ Block 1 ── [Conv 32  → BN → ReLU] ×2 → MaxPool
   ├─ Block 2 ── [Conv 64  → BN → ReLU] ×2 → MaxPool
   ├─ Block 3 ── [Conv 128 → BN → ReLU] ×2 → MaxPool
   │
   ├─ GlobalAveragePooling2D
   ├─ Dropout (0.4) → Dense (128, ReLU) → Dropout (0.3)
   └─ Dense (10)  → logits  (softmax applied at prediction time)
```

> **Why Global Average Pooling?** Compared with `Flatten + Dense(128)`, it cuts the parameter count from ~1.1M to ~0.3M, which means less overfitting and a smaller model file.

---

## 🚀 Quickstart

### Option A: Kaggle (easiest)

1. Upload `eurosat.ipynb` to [Kaggle](https://www.kaggle.com/code).
2. Turn on **GPU** and **Internet** in the notebook settings (the dataset is fetched with `kagglehub`).
3. Click **Run All**. ☕

### Option B: Local

```bash
git https://github.com/Moeed-S-E/eurosat.git
cd eurosat

pip install tensorflow kagglehub scikit-learn matplotlib numpy

python eurosat.py
```

> 🔑 Local runs need Kaggle credentials for `kagglehub`. See the [kagglehub docs](https://github.com/Kaggle/kagglehub).

---

## 📦 Dataset

[**EuroSAT (Sentinel-2)**](https://www.kaggle.com/datasets/raoofnaushad/eurosat-sentinel2-dataset): 27,000 labelled RGB tiles at 64×64 px, covering 10 land-use and land-cover classes across Europe.

> Helber, P., Bischke, B., Dengel, A., Borth, D. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.

---

## 📈 Results

> Run the notebook and drop your numbers in here. 👇
Test loss: 0.1979 | Test accuracy: 

| Metric | Value |
|---|---|
| Test accuracy | 93.63% |
| Test loss | 0.1979 |
| Epochs run (early-stopped) | 35/40 |

<!-- Tip: save your plots from the notebook and embed them:
![Training curves](assets/curves.png)
![Confusion matrix](assets/confusion_matrix.png)
-->

---

## 🧪 Pipeline at a Glance

```text
kagglehub download
      ↓
parse labels from filenames  →  sanity-check class counts
      ↓
stratified split  (70 train / 15 val / 15 test)
      ↓
tf.data  (decode → resize → cache → shuffle → batch → prefetch)
      ↓
CNN + augmentation  →  EarlyStopping · ReduceLROnPlateau · ModelCheckpoint
      ↓
test evaluation  →  classification report · confusion matrix · single-image demo
      ↓
model.save("eurosat_cnn_model.keras")
```

---

## 📁 Repository Structure

```text
.
├── Dataset/
├── eurosat.ipynb   # full notebook, ready for Kaggle
└── README.md
```

---

## 🔧 Configuration

All knobs are at the top of the script:

| Variable | Default | What it does |
|---|---|---|
| `IMG_SIZE` | `(64, 64)` | Input resolution |
| `BATCH_SIZE` | `64` | Batch size |
| `EPOCHS` | `40` | Max epochs (early stopping usually ends sooner) |
| `SEED` | `123` | Reproducible splits and shuffling |

---

## 🛣️ Roadmap

- [ ] Transfer learning (EfficientNet / ResNet) for a stronger baseline
- [ ] Test-time augmentation
- [ ] Grad-CAM heatmaps to see what the model looks at
- [ ] Multispectral (13-band) version of EuroSAT
- [ ] Export to TFLite for edge deployment

---

## 🤝 Contributing

Issues and pull requests are welcome. If you have a trick that squeezes out more accuracy, open a PR.

## 📄 License

Choose a license (e.g. [MIT](https://choosealicense.com/licenses/mit/)) and add a `LICENSE` file. The dataset has its own terms; check the original EuroSAT source.

---

<div align="center">

⭐ **If this helped you, give it a star!** ⭐

</div>
