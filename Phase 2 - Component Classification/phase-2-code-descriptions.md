# Phase 2 – Component Classification

This folder contains scripts for **component-level defect detection**. Unlike Phase 1, which classifies entire boards, Phase 2 leverages **fiducial alignment and CAD data (DXF/STEP)** to locate and crop individual components (e.g., switches). Cropped images are then used to train a **YOLOv8 object detector**.

---

## File Descriptions

- **`panel_to_pcbs_with_step_dxf_yolo.py`**  
  End-to-end pipeline for extracting and aligning PCBs from a panel:
  1. Detect fiducials  
  2. Pair diagonals → isolate boards  
  3. Parse DXF outline and STEP components  
  4. Warp each board into a square reference frame  
  5. Extract component crops (switches)  
  6. Export YOLO-format images + labels  

- **`pattern_match_fiducials_patched.py`**  
  Standalone fiducial detection script using template matching with non-maximum suppression. Outputs fiducial coordinates for alignment.

- **`relocate_switches_yolo.py`**  
  Maps STEP/DXF component bounding boxes onto warped PCB images.  
  - Corrects orientation of bottom-row boards (180° flip)  
  - Converts bounding boxes into YOLO annotation format  

- **`yolo_switches.ipynb`**  
  Jupyter notebook for training YOLOv8 on cropped switches, visualizing predictions, and evaluating detection accuracy.

- **`phase-2-requirements.txt`**  
  Dependencies for this phase: OpenCV, ezdxf, pythonocc-core, ultralytics, matplotlib, numpy, etc.

- **`c`** *(removed)*  
  This file was an empty placeholder and has been deleted.

---

## Usage

1. Install dependencies:
   ```bash
   pip install -r phase-2-requirements.txt
2. Run fiducial detection and board extraction
   ```bash
   python panel_to_pcbs_with_step_dxf_yolo.py
3. Train and test YOLO using yolo_switches.ipynb cells
