# Automated Optical Inspection (AOI) for PCB Defect Detection

**MEng Research Project – MIE8888Y, University of Toronto**  
Author: *Tiffany Costas*  
Supervisor: *Professor Janet Lam*  
Partner: [Ploopy Corporation](https://ploopy.co/)

---

## 📖 Overview

This project develops an **Automated Optical Inspection (AOI)** pipeline for **Printed Circuit Board (PCB)** defect detection. The system combines classical computer vision, deep learning, and CAD-driven alignment to deliver a reproducible, open-source solution for PCB quality control.

- **Phase 1 – Board-Level Classification**  
  - Whole-board pass/fail classification using **ResNet18 (transfer learning)**.  
  - Tested on three Ploopy boards: **ADEPT**, **Trackpad**, and **Thumb R1**.  
  - Data augmentation: rotations, scaling, brightness/contrast jitter, pin-cushion distortion.  

- **Phase 2 – Component-Level Detection**  
  - **Fiducial detection** + **homography alignment** with CAD files (DXF/STEP).  
  - Automatic cropping of component regions (demonstrated on **switches**).  
  - **YOLOv8** for real-time detection of *good_switch* vs *bad_switch*.  

---

## 🛠️ Installation

### Requirements
- Python 3.10 (Anaconda recommended)  
- CUDA-enabled GPU (optional but recommended for training)  

### Setup

```bash
# Clone repository
git clone https://github.com/<your-username>/MEng-PROJECT.git
cd pcb-aoi

# Create and activate conda environment
conda create -n pcb_aoi python=3.10 -y
conda activate pcb_aoi

# Install dependencies
pip install -r phase-1-requirements.txt
pip install -r phase-2-requirements.txt
