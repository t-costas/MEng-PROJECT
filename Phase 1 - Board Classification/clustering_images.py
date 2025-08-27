import os
import shutil
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import pandas as pd

reference_dir = "reference"
input_dir = "dataset"
output_dir = "dataset_sorted"
csv_log_path = "classification_log.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def extract_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).squeeze().cpu().numpy()

# Load reference embeddings
reference_embeddings = []
reference_labels = []

for fname in os.listdir(reference_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        fpath = os.path.join(reference_dir, fname)
        label = "_".join(fname.split("_")[:2])  # e.g., board1_pass
        emb = extract_embedding(fpath)
        reference_embeddings.append(emb)
        reference_labels.append(label)

reference_embeddings = np.array(reference_embeddings)

# Match and sort input images
log_data = []

def classify_and_sort(split):
    for label in ["pass", "fail"]:
        src_dir = os.path.join(input_dir, split, label)
        if not os.path.exists(src_dir):
            continue
        for fname in tqdm(os.listdir(src_dir), desc=f"{split}/{label}"):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src_path = os.path.join(src_dir, fname)
            try:
                emb = extract_embedding(src_path).reshape(1, -1)
                dists = cosine_distances(emb, reference_embeddings)[0]
                best_match = reference_labels[np.argmin(dists)]
                dest_dir = os.path.join(output_dir, split, best_match)
                os.makedirs(dest_dir, exist_ok=True)
                new_name = f"{best_match}_{fname}"
                dest_path = os.path.join(dest_dir, new_name)
                shutil.copy2(src_path, dest_path)
                log_data.append({
                    "split": split,
                    "original_path": src_path,
                    "predicted_label": best_match,
                    "destination_path": dest_path
                })
            except Exception as e:
                print(f"Error with {fname}: {e}")

classify_and_sort("train")
classify_and_sort("val")

pd.DataFrame(log_data).to_csv(csv_log_path, index=False)
print("Done. Images sorted and renamed.")
print(f"Log saved at: {csv_log_path}")
