import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
from glob import glob
from tqdm import tqdm

# === Config ===
base_dir = "dataset"
sets = [("train", 0.6), ("val", 0.2)]
target_total = 1000  # Total number of images (across all sets/classes)
max_rotation = 5  # degrees
zoom_range = (0.9, 1.1)  # zoom out, zoom in
output_image_format = "jpg"

# === Augmentations ===
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def zoom_image(img, zoom_factor):
    h, w = img.shape[:2]
    nh, nw = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(img, (nw, nh))

    if zoom_factor < 1.0:  # pad
        pad_h = (h - nh) // 2
        pad_w = (w - nw) // 2
        result = cv2.copyMakeBorder(resized, pad_h, h - nh - pad_h, pad_w, w - nw - pad_w, cv2.BORDER_REFLECT)
    else:  # crop
        start_h = (nh - h) // 2
        start_w = (nw - w) // 2
        result = resized[start_h:start_h + h, start_w:start_w + w]
    return result

def adjust_brightness_contrast(img, brightness=1.0, contrast=1.0):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def augment(img):
    angle = random.uniform(-max_rotation, max_rotation)
    zoom = random.uniform(*zoom_range)
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)

    img = rotate_image(img, angle)
    img = zoom_image(img, zoom)
    img = adjust_brightness_contrast(img, brightness, contrast)
    return img

def augment_class(class_dir, target_count):
    os.makedirs(class_dir, exist_ok=True)
    image_paths = glob(os.path.join(class_dir, "*"))
    base_images = [cv2.imread(p) for p in image_paths if cv2.imread(p) is not None]

    current_count = len(base_images)
    needed = max(0, target_count - current_count)

    print(f" - {class_dir}: {current_count} → {target_count} (adding {needed})")

    for i in tqdm(range(needed)):
        src_idx = i % current_count
        img = base_images[src_idx]
        aug_img = augment(img)
        out_path = os.path.join(class_dir, f"aug_{i}.{output_image_format}")
        cv2.imwrite(out_path, aug_img)

def main():
    for split_name, split_ratio in sets:
        print(f"\nProcessing: {split_name.upper()}")
        target_per_class = int((target_total * split_ratio) / 2)  # half for pass, half for fail
        for cls in ["pass", "fail"]:
            class_dir = os.path.join(base_dir, split_name, cls)
            augment_class(class_dir, target_per_class)

    print("\nDataset augmented and balanced with 60/20/20 split.")

if __name__ == "__main__":
    main()
