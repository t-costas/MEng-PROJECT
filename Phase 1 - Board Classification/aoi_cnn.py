import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import ResNet18_Weights
import datetime

print("Starting...")

data_dir = "dataset_sorted"
batch_size = 8
num_epochs = 10
learning_rate = 6e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

train_labels = [label for _, label in train_dataset]
val_labels = [label for _, label in val_dataset]
print("Train class distribution:", Counter(train_labels))
print("Val class distribution:", Counter(val_labels))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_classes = len(train_dataset.classes)  # Automatically detects total classes "board0_pass"
print(num_classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
misclassified_images = []
misclassified_labels = []
misclassified_preds = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    val_loss = 0.0
    correct_val, total_val = 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified_images.append(inputs[i].cpu())
                    misclassified_labels.append(labels[i].cpu().numpy())
                    misclassified_preds.append(preds[i].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")
# === Per-Board-Type Accuracy ===
from collections import defaultdict

board_correct = defaultdict(int)
board_total = defaultdict(int)

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for pred, label in zip(preds, labels):
            true_class = train_dataset.classes[label]
            board = "_".join(true_class.split("_")[:1])  # 'board0_pass' → 'board0'
            board_total[board] += 1
            if pred == label:
                board_correct[board] += 1

print("\n📊 Per-Board-Type Validation Accuracy:")
for board in sorted(board_total.keys()):
    acc = board_correct[board] / board_total[board]
    print(f"{board}: {acc:.2%} ({board_correct[board]}/{board_total[board]})")

# === Save Model ===
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
timestamped_path = f"model_{current_date}.pth"
latest_path = "model_latest.pth"

torch.save(model.state_dict(), timestamped_path)
torch.save(model.state_dict(), latest_path)

print(f"Model saved at {timestamped_path}")
print(f"Also saved latest model to {latest_path}")


'''# === Plot Loss & Accuracy ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(train_accuracies, label="Train Accuracy", marker='o')
plt.plot(val_accuracies, label="Val Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Confusion Matrix ===
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix on Validation Set")
plt.tight_layout()
plt.show()

for i in range(min(5, len(misclassified_images))):
    img = transforms.ToPILImage()(misclassified_images[i])
    plt.imshow(img)
    plt.title(f"True: {misclassified_labels[i]}, Pred: {misclassified_preds[i]}")
    plt.show()
'''