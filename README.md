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
```

## Phase 1 – Board-Level Dataset (ADEPT, Trackpad, Thumb R1)
Download from Google Drive: [📥 Phase 1 Dataset](https://drive.google.com/file/d/1WnA2TPdBWIgrhWJEhPM4kcZS8EfwZ4Vh/view?usp=sharing)

The full dataset for Phase 1 is hosted externally to keep this repository lightweight.  

## Phase 2 Help (`panel_to_pcbs_with_step_dxf_yolo.py`)

This script runs the **end-to-end Phase 2 pipeline**:

1. Detects fiducials on the PCB panel.  
2. Computes NE/SW diagonal pairs → identifies individual boards.  
3. Loads **DXF outline** (for board size/position) and **STEP file** (for component footprints).  
4. Warps each PCB image to a square reference size.  
5. Extracts component crops (switches).  
6. Generates **YOLOv8** directory and exports bounding boxes + labels.  
7. Saves warped boards, overlays, and cropped components.

#### Example (Windows PowerShell)

```powershell
& "C:\Users\costas\anaconda3\envs\occ-env\python.exe" `
  "C:\Users\costas\OneDrive\Desktop\MEng PROJECT\panel_to_pcbs_with_step_dxf_yolo.py" `
  --imageFilepath "C:\Users\costas\OneDrive\Desktop\MEng PROJECT\images\board0_pass_nodefects_6.png" `
  --step_path      "C:\Users\costas\OneDrive\Desktop\MEng PROJECT\Adept\Adept.STEP" `
  --dxf_path       "C:\Users\costas\OneDrive\Desktop\MEng PROJECT\Adept\Adept.DXF" `
  --dxf_outline_layer "Mechanical2" `
  --ne_x_mm 22.4 --ne_y_mm -20.4 `
  --sw_x_mm -18.4 --sw_y_mm 30.0 `
  --pcb_out_size 900 `
  --outputDirectory "C:\Users\costas\OneDrive\Desktop\MEng PROJECT\outputs" `
  --save_component_crops `
  --dxf_y_up `
  --export_yolo `
  --yolo_dir "C:\Users\costas\OneDrive\Desktop\MEng PROJECT\outputs\switches_yolo"

