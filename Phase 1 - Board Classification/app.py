import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import os
from datetime import datetime
import numpy as np
import pandas as pd

# Board name map
BOARD_NAME_MAP = {
    "board0": "ADEPT",
    "board1": "Trackpad",
    "board2": "Thumb R1"
}

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(get_all_classes()))
    model.load_state_dict(torch.load(r"C:\Users\costas\OneDrive\Desktop\MEng PROJECT\Phase 1 - Board Classification\model_weights\model_latest.pth", map_location="cpu")) # MODIFY TO YOUR PATH
    model.eval()
    return model

def get_all_classes():
    class_dir = os.path.join(r"C:\Users\costas\OneDrive\Desktop\MEng PROJECT\dataset_sorted", "train")
    if not os.path.exists(class_dir):
        return []
    return sorted([d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))])

def get_next_board_id():
    class_names = get_all_classes()
    ids = [int(name[5:].split("_")[0]) for name in class_names if name.startswith("board") and name[5:].split("_")[0].isdigit()]
    return f"board{max(ids, default=-1) + 1}"

def extract_board_name(label):
    board = "_".join(label.split("_")[:1])
    return BOARD_NAME_MAP.get(board, board)

def get_mapped_board_choices():
    boards = sorted(set([cls.split("_")[0] for cls in get_all_classes()]))
    return {BOARD_NAME_MAP.get(b, b): b for b in boards}

model = load_model()
class_names = get_all_classes()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize session state
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "last_saved_path" not in st.session_state:
    st.session_state.last_saved_path = None
if "nav_click" not in st.session_state:
    st.session_state.nav_click = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

st.title("PCB AOI Pass/Fail Classifier")

# Clear predictions
if st.button("Clear Current Predictions"):
    st.session_state.predictions = []
    st.session_state.selected_idx = 0
    st.session_state.nav_click = None
    st.session_state.uploaded_files = []
    st.rerun()

# Upload new images
if not st.session_state.predictions:
    uploaded_files = st.file_uploader(
        "Upload one or more PCB images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_pred"
    )
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
else:
    uploaded_files = st.session_state.uploaded_files

# Generate predictions
if uploaded_files and not st.session_state.predictions:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB").copy()
        tensor_img = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.softmax(outputs[0], dim=0)

        pred_idx = int(probs.argmax())
        pred_label = class_names[pred_idx]
        board_type = extract_board_name(pred_label)
        status = pred_label.split("_")[1].upper()
        confidence = float(probs[pred_idx])

        st.session_state.predictions.append({
            "image": img,
            "filename": file.name,
            "board_type": board_type,
            "status": status,
            "confidence": confidence,
            "probs": probs.numpy(),
            "pred_label": pred_label
        })
    st.session_state.selected_idx = 0

predictions = st.session_state.predictions

# Navigation logic
if predictions and len(predictions) > 1:
    if st.session_state.nav_click == "prev" and st.session_state.selected_idx > 0:
        st.session_state.selected_idx -= 1
    if st.session_state.nav_click == "next" and st.session_state.selected_idx < len(predictions) - 1:
        st.session_state.selected_idx += 1
    st.session_state.nav_click = None

# Display selected image and prediction
if predictions:
    selected_idx = st.session_state.selected_idx
    selected = predictions[selected_idx]

    st.image(
        selected["image"],
        caption=f"Image {selected_idx + 1} of {len(predictions)}",
        use_container_width=True
    )

    # Nav buttons
    if len(predictions) > 1:
        nav_spacer1, nav_buttons, nav_spacer2 = st.columns([1, 6, 1])
        with nav_buttons:
            left_col, right_col = st.columns([1, 1])

            with left_col:
                if st.button("Previous", key=f"nav_prev_{selected_idx}", use_container_width=True):
                    st.session_state.nav_click = "prev"
                    st.rerun()
            with right_col:
                if st.button("Next", key=f"nav_next_{selected_idx}", use_container_width=True):
                    st.session_state.nav_click = "next"
                    st.rerun()

    status_color = "green" if selected["status"] == "PASS" else "red"
    st.markdown(
        f"<h2>Prediction: <span style='color:{status_color};'>{selected['status']}</span></h2>",
        unsafe_allow_html=True
    )
    st.write(f"**Filename:** {selected['filename']}")
    st.write(f"**Board Type:** {selected['board_type']}")
    st.write(f"**Confidence:** {selected['confidence']:.2%}")

    st.subheader("Class Probabilities")
    for idx, class_name in enumerate(class_names):
        st.write(f"{class_name}: {selected['probs'][idx]:.3f}")

    # Correction section
    st.subheader("Was this prediction correct?")
    correction_needed = st.radio("Prediction correct?", ["Yes", "No"], horizontal=True, key=f"correct_{selected_idx}")

    default_board = selected["pred_label"].split("_")[0]
    default_status = selected["pred_label"].split("_")[1]

    if correction_needed == "No":
        st.markdown("### Relabel the Image")
        new_board = st.checkbox("New board type", key=f"correct_new_{selected_idx}")
        status_choice = st.selectbox(
            "Correct label (pass/fail):",
            ["pass", "fail"],
            index=0 if default_status == "pass" else 1,
            key=f"correct_status_{selected_idx}"
        )

        if new_board:
            new_board_id = get_next_board_id()
            label = f"{new_board_id}_{status_choice}"
        else:
            board_choices_map = get_mapped_board_choices()
            board_choice_display = st.selectbox(
                "Select correct board type:",
                list(board_choices_map.keys()),
                index=list(board_choices_map.values()).index(default_board),
                key=f"correct_board_{selected_idx}"
            )
            board_choice = board_choices_map[board_choice_display]
            label = f"{board_choice}_{status_choice}"

        if st.button(f"Correct and Save Image {selected_idx}"):
            save_path = os.path.join(r"C:\Users\costas\OneDrive\Desktop\MEng PROJECT\dataset_sorted", "train", label)
            os.makedirs(save_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{label}_{timestamp}.jpg"
            full_save_path = os.path.join(save_path, filename)
            selected["image"].save(full_save_path)
            st.session_state.last_saved_path = full_save_path
            st.success(f"Corrected and saved to `{full_save_path}`")

        if st.session_state.last_saved_path:
            if st.button("Undo Last Correction"):
                try:
                    os.remove(st.session_state.last_saved_path)
                    st.success(f"Undid correction: Deleted `{st.session_state.last_saved_path}`")
                    st.session_state.last_saved_path = None
                except Exception as e:
                    st.error(f"Failed to undo: {e}")

    # Add to training set
    st.subheader("Add This Image to Training Set")
    new_board = st.checkbox("New board type", key=f"add_new_{selected_idx}")
    status_choice = st.selectbox(
        "Label (pass/fail):",
        ["pass", "fail"],
        index=0 if default_status == "pass" else 1,
        key=f"status_{selected_idx}"
    )

    if new_board:
        label = f"{get_next_board_id()}_{status_choice}"
    else:
        board_choices_map = get_mapped_board_choices()
        board_choice_display = st.selectbox(
            "Select board type:",
            list(board_choices_map.keys()),
            index=list(board_choices_map.values()).index(default_board),
            key=f"board_{selected_idx}"
        )
        board_choice = board_choices_map[board_choice_display]
        label = f"{board_choice}_{status_choice}"

    if st.button(f"Save Image {selected_idx} to Training Set"):
        save_path = os.path.join(r"C:\Users\costas\OneDrive\Desktop\MEng PROJECT\dataset_sorted", "train", label)
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.jpg"
        selected["image"].save(os.path.join(save_path, filename))
        st.success(f"Saved to `{save_path}/{filename}`")

# Export to CSV
if predictions:
    if st.button("Export Predictions to CSV"):
        data = []
        for p in predictions:
            row = {
                "filename": p["filename"],
                "board_type": p["board_type"],
                "status": p["status"],
                "confidence": p["confidence"]
            }
            for idx, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = p["probs"][idx]
            data.append(row)
        df = pd.DataFrame(data)
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("Download CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

# Retrain model
st.subheader("Retrain Model")
if st.button("Start Retraining"):
    with st.spinner("Retraining model... this may take a minute."):
        os.system("python aoi_cnn.py")
        st.success("Retraining complete!")
        st.cache_resource.clear()
        model = load_model()
        st.success("Model reloaded with updated weights.")

st.info("Model: ResNet18. Supports batch upload, prediction, correction, undo, retraining, and CSV export.")
