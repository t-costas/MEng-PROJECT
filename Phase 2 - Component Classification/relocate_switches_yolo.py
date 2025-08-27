# relocate_switches_yolo.py
# Create position-randomized YOLO training/validation images from an existing dataset.
# For each image in the selected splits, this script:
#   1) reads the YOLO labels (assumes 6 switches per PCB image),
#   2) extracts switch crops,
#   3) masks the original locations,
#   4) creates N augmented variants by pasting the crops at random positions,
#   5) writes new images and corresponding YOLO .txt labels (.augK suffix).
#
# NOTE: Augmenting the *validation* split reduces the realism of your metrics.
#       It’s fine for feasibility checks; for final reporting, prefer a non-aug val set.

from pathlib import Path
from PIL import Image, ImageDraw
import random

# ---------------- CONFIG ----------------
DATA_ROOT = Path(r"C:/Users/costas/OneDrive/Desktop/MEng PROJECT/outputs_yolo/switches_yolo")

# How many augmented copies to create per split
AUG_PER_SPLIT = {
    "train": 6,   # e.g., 6 variants per train image
    "val":   3,   # e.g., 3 variants per val image
    # you can add "test": 0 if you want to skip
}

# Paste scale jitter and placement attempts
SCALE_MIN, SCALE_MAX = 0.9, 1.2
JITTER_TRIES = 200

# Color to cover the original switch regions
MASK_COLOR = (128, 128, 128)

# Acceptable image extensions
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# RNG seed for reproducibility
RANDOM_SEED = 42
# ----------------------------------------


def read_yolo_labels(lab_file: Path, W: int, H: int):
    """Read YOLO labels -> list of (cid, x0, y0, x1, y1) in pixel coords."""
    boxes = []
    if not lab_file.exists():
        return boxes
    text = lab_file.read_text().strip().splitlines()
    for line in text:
        if not line.strip():
            continue
        cid, cx, cy, w, h = line.split()
        cid = int(cid); cx = float(cx); cy = float(cy); w = float(w); h = float(h)
        x0 = (cx - w/2) * W; y0 = (cy - h/2) * H
        x1 = (cx + w/2) * W; y1 = (cy + h/2) * H
        boxes.append((cid, x0, y0, x1, y1))
    return boxes


def write_yolo_labels(lab_file: Path, boxes_xyxy, W: int, H: int):
    """Write list of (cid, x0, y0, x1, y1) as YOLO normalized lines."""
    lines = []
    for cid, x0, y0, x1, y1 in boxes_xyxy:
        x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W - 1, x1))
        y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H - 1, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        cx = (x0 + x1) / 2.0 / W
        cy = (y0 + y1) / 2.0 / H
        w  = (x1 - x0) / W
        h  = (y1 - y0) / H
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lab_file.write_text("\n".join(lines) + ("\n" if lines else ""))


def crop_region(im: Image.Image, x0, y0, x1, y1):
    return im.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))


def mask_boxes(im: Image.Image, boxes, color=MASK_COLOR):
    im2 = im.copy()
    draw = ImageDraw.Draw(im2)
    for _, x0, y0, x1, y1 in boxes:
        draw.rectangle([x0, y0, x1, y1], fill=color)
    return im2


def paste_with_scale(canvas: Image.Image, crop_im: Image.Image, scale: float, x: int, y: int):
    W, H = canvas.size
    w0, h0 = crop_im.size
    w = max(2, int(round(w0 * scale)))
    h = max(2, int(round(h0 * scale)))
    crop_rs = crop_im.resize((w, h), Image.BILINEAR)
    x = int(round(max(0, min(W - w, x))))
    y = int(round(max(0, min(H - h, y))))
    canvas.paste(crop_rs, (x, y))
    return x, y, x + w, y + h


def process_split(split_name: str, n_aug: int):
    if n_aug <= 0:
        print(f"[{split_name}] n_aug=0 → skipping.")
        return
    img_dir = DATA_ROOT / "images" / split_name
    lab_dir = DATA_ROOT / "labels" / split_name
    if not img_dir.exists():
        print(f"[{split_name}] No folder: {img_dir} → skipping.")
        return

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"[{split_name}] Found {len(imgs)} images in {img_dir}")

    for img_path in imgs:
        lab_path = lab_dir / f"{img_path.stem}.txt"
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        boxes = read_yolo_labels(lab_path, W, H)
        if len(boxes) == 0:
            # no labels → skip
            continue

        # extract crops (once per source image)
        crops = []
        for cid, x0, y0, x1, y1 in boxes:
            crops.append((cid, crop_region(im, x0, y0, x1, y1)))

        # base canvas with original switch locations masked out
        base = mask_boxes(im, boxes, color=MASK_COLOR)

        # generate augmented variants
        for k in range(n_aug):
            canvas = base.copy()
            new_boxes = []
            for cid, crop in crops:
                placed = False
                for _ in range(JITTER_TRIES):
                    sc = random.uniform(SCALE_MIN, SCALE_MAX)
                    w0, h0 = crop.size
                    ww = max(2, int(round(w0 * sc)))
                    hh = max(2, int(round(h0 * sc)))
                    x = random.randint(0, max(0, W - ww))
                    y = random.randint(0, max(0, H - hh))
                    x0, y0, x1, y1 = paste_with_scale(canvas, crop, sc, x, y)
                    new_boxes.append((cid, x0, y0, x1, y1))
                    placed = True
                    break
                if not placed:
                    # fallback: paste without scaling centered
                    x0 = max(0, min(W - w0, (W - w0)//2))
                    y0 = max(0, min(H - h0, (H - h0)//2))
                    canvas.paste(crop, (x0, y0))
                    new_boxes.append((cid, x0, y0, x0 + w0, y0 + h0))

            out_img = img_dir / f"{img_path.stem}.aug{k}.png"
            out_lab = lab_dir / f"{img_path.stem}.aug{k}.txt"
            canvas.save(out_img)
            write_yolo_labels(out_lab, new_boxes, W, H)

    print(f"[{split_name}] Augmentation complete: +{n_aug} variants per image.")


def main():
    random.seed(RANDOM_SEED)
    for split, n_aug in AUG_PER_SPLIT.items():
        process_split(split, n_aug)


if __name__ == "__main__":
    main()
