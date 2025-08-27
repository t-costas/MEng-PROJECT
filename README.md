Automated Optical Inspection (AOI) for PCB Defect Detection

MEng Research Project (MIE8888Y, University of Toronto)
Author: Tiffany Costas
Supervisor: Professor Janet Lam
Partner: Ploopy Corporation

---

This project develops an Automated Optical Inspection (AOI) pipeline for Printed Circuit Board (PCB) defect detection, combining:

Phase 1 – Board-Level Classification:

Pass/fail classification of whole PCB images using ResNet18 (transfer learning).

Data augmentation: rotations, scaling, brightness/contrast jitter, pin-cushion distortion.

Evaluated on three board types: ADEPT, Trackpad, and Thumb R1.

Phase 2 – Component-Level Detection:

Fiducial detection and homography alignment to CAD design files (DXF/STEP).

Automatic extraction of component bounding boxes (currently demonstrated on switches).

Real-time defect localization using YOLOv8 with good_switch vs bad_switch classes.

This hybrid approach demonstrates the feasibility of integrating classical computer vision, deep learning, and CAD-driven alignment into a reproducible open-source inspection framework.
