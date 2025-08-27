import os
import shutil
import random
from pathlib import Path

# Paths 
source_dir = Path("PCB Images")
output_dir = Path("dataset")
categories = ["pass", "fail"]

# Collect and label images 
image_paths = list(source_dir.glob("*.png"))
labeled_images = [
    (path, "pass" if path.name.startswith("nodefects") else "fail")
    for path in image_paths
    if "_MG_" not in path.name
]

random.shuffle(labeled_images)
N = len(labeled_images)

train_end = int(0.6 * N)
val_end   = int(0.8 * N)

train_data = labeled_images[:train_end]
val_data   = labeled_images[train_end:val_end]
test_data  = labeled_images[val_end:]

# Helper to copy images into subset folders 
def organize(data, subset):
    for path, label in data:
        target_dir = output_dir / subset / label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target_dir / path.name)

# Organize into train/val/test 
organize(train_data, "train")
organize(val_data,   "val")
organize(test_data,  "test")

print(f"✅ Done! {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")
