# Phase 1 – Board Classification

This folder contains scripts for **board-level pass/fail classification** of printed circuit boards (PCBs) using a convolutional neural network (ResNet18). The model is trained on images of entire boards and classifies them into defect / no-defect categories.

---

## File Descriptions

- **`aoi_augmentations.py`**  
  Implements custom image augmentation functions (rotation, scaling, brightness jitter, pin-cushion distortion, etc.) to improve dataset variability and model robustness.

- **`aoi_cnn.py`**  
  Core training script for the classification model.  
  - Loads and preprocesses dataset  
  - Builds a ResNet18 classifier (transfer learning)  
  - Trains the model and saves checkpoints  

- **`app.py`**  
  Streamlit web app for PCB classification.  
  - Upload PCB images  
  - Run predictions (pass/fail) with model confidence  
  - Export results to CSV  

- **`clustering_images.py`**  
  Utility script for clustering dataset images (e.g., KMeans). Useful for detecting mislabeled or duplicate images and analyzing dataset distribution.

- **`data-cleaning.ipynb`**  
  Jupyter notebook for dataset preparation. Handles preprocessing, resizing, renaming, and cleaning of raw PCB images before training.

- **`sorting_dataset.py`**  
  Automates dataset organization into the required directory structure (`train/`, `val/`, `test/`) with subfolders by board type (e.g., ADEPT, Trackpad, Thumb R1).

- **`phase-1-requirements.txt`**  
  Python dependencies for this phase: PyTorch, torchvision, Streamlit, matplotlib, numpy, etc.

- **`model_weights/`**  
  Stores trained model weights (`.pth` files). Example: `model_latest.pth`.

- **`training results/`**  
  Stores experiment outputs including training logs, loss/accuracy curves, and confusion matrices.

---

## Usage

1. Install dependencies:
   ```bash
   pip install -r phase-1-requirements.txt
2. Train the model to update weights
   ```bash
   python aoi_cnn.py
3. Run the web app
   ```bash
   streamlit run app.py
