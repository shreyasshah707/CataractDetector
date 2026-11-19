"""
EDD v2 — Streamlit Research App: Two-Stage Cataract Diagnostic System
----------------------------------------------------------------------
Interactive research tool for evaluating eye photos.

Pipeline:
  - Stage 1 (Luminance + SVM): Optical lens ROI extraction & normal vs cataract detection
  - Stage 2 (EfficientNet-B0): Severity grading (Immature vs Mature) for detected cataracts
  - Explainability: Lens ROI Segmentation & Grad-CAM Attention Heatmaps

Usage:
    streamlit run app.py
"""

import os
os.environ["KERAS_HOME"] = r"D:\Projects\EDD\v2\.keras_cache"
os.environ["MPLCONFIGDIR"] = r"D:\Projects\EDD\v2\.mpl_cache"

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import cv2
import streamlit as st
import joblib
from PIL import Image

import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
STAGE1_SVM_PATH = "models/stage1/stage1_svm.pkl"
STAGE1_PARAMS_PATH = "models/stage1/stage1_params.json"
B0_SEVERITY_PATH = "models/efficientnet_b0_best.h5"
B0_BINARY_PATH = "models/checkpoints_b0_binary/best.h5"

IMG_SIZE = (224, 224)
RESULTS_DIR = Path("results/streamlit")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="EDD v2 — Cataract Diagnostic System",
    page_icon="👁️",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    svm_model = joblib.load(STAGE1_SVM_PATH)
    with open(STAGE1_PARAMS_PATH, "r", encoding="utf-8") as f:
        stage1_params = json.load(f)
    b0_severity = keras.models.load_model(B0_SEVERITY_PATH)
    b0_binary = keras.models.load_model(B0_BINARY_PATH)
    return svm_model, stage1_params, b0_severity, b0_binary


# ---------------------------------------------------------------------------
# Feature Extraction (Stage 1)
# ---------------------------------------------------------------------------
def extract_luminance(img_rgb, img_size=(224, 224)):
    img_uint8 = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LINEAR)
    img_filtered = cv2.medianBlur(img_uint8, 5)

    gray = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    opened = cv2.morphologyEx(grad, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    markers = markers.astype(np.uint8)
    markers = cv2.connectedComponents(markers)[1] + 1
    markers[thresh == 0] = 0
    markers = cv2.watershed(img_filtered, markers)
    lens_mask = (markers > 1).astype(np.uint8) * 255

    img_float = img_filtered.astype(np.float32) / 255.0
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    BV = 0.299 * R + 0.587 * G + 0.114 * B
    luminance = np.power(2.0, BV) / 0.0929

    lens_mask_bool = lens_mask > 0
    if np.sum(lens_mask_bool) == 0:
        h, w = luminance.shape
        center_mask = np.zeros((h, w), dtype=bool)
        center_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        lens_mask_bool = center_mask
        lens_mask = center_mask.astype(np.uint8) * 255

    mean_lum = float(np.mean(luminance[lens_mask_bool]))
    std_lum = float(np.std(luminance[lens_mask_bool]))

    overlay = img_uint8.copy()
    overlay[lens_mask > 0] = (overlay[lens_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    return mean_lum, std_lum, overlay, lens_mask


# ---------------------------------------------------------------------------
# Grad-CAM Helpers
# ---------------------------------------------------------------------------
def compute_gradcam(model, img_array, class_idx):
    try:
        base_model = model.get_layer("efficientnetb0")
        last_conv = base_model.get_layer("top_conv")
        sub_base = keras.Model(inputs=base_model.inputs, outputs=[last_conv.output, base_model.output])

        classifier_layers = []
        found_base = False
        for l in model.layers:
            if found_base:
                classifier_layers.append(l)
            if l.name == "efficientnetb0":
                found_base = True

        with tf.GradientTape() as tape:
            conv_out, pooled_out = sub_base(img_array)
            tape.watch(conv_out)
            x = pooled_out
            for l in classifier_layers:
                x = l(x, training=False)
            loss = x[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        st.warning(f"Grad-CAM note: {e}")
        return None


def overlay_gradcam(orig_pil, heatmap, img_size=(224, 224)):
    if heatmap is None:
        return None
    orig = orig_pil.convert("RGB").resize(img_size, Image.Resampling.BILINEAR)
    orig_arr = np.array(orig).astype(np.float32) / 255.0

    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], img_size).numpy()[..., 0]
    jet = matplotlib.colormaps["jet"]
    heatmap_colored = jet(heatmap_resized)[..., :3]

    overlay = 0.6 * heatmap_colored + 0.4 * orig_arr
    overlay = np.clip(overlay, 0, 1)
    overlay_img = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay_img)


# ---------------------------------------------------------------------------
# App Interface
# ---------------------------------------------------------------------------
st.title("👁️ EDD v2 — Cataract Diagnostic & Severity Grading System")
st.markdown(
    """
    Research tool implementing the **Two-Stage Architecture**:
    1. **Stage 1 (Optical Lens Luminance + SVM):** Segments the crystalline lens ROI to detect **Normal vs. Cataract** (Askarian et al. 2021).
    2. **Stage 2 (EfficientNet-B0 Severity Grader):** If Cataract is detected, grades severity as **Immature vs. Mature**.
    """
)

# Load Models
try:
    svm_model, stage1_params, b0_severity, b0_binary = load_all_models()
    st.sidebar.success("✅ Models loaded successfully")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.header("Evaluation Mode")
eval_mode = st.sidebar.radio(
    "Choose evaluation pipeline:",
    ["Full Two-Stage Pipeline (Recommended)", "Individual Models Diagnostic"]
)

show_roi = st.sidebar.checkbox("Show Lens ROI Segmentation Mask", value=True)
show_cam = st.sidebar.checkbox("Show Grad-CAM Attention Heatmaps", value=True)

st.sidebar.divider()
st.sidebar.caption("Benchmark Metrics on Real Photos ($N=17$):")
st.sidebar.caption("• Two-Stage Accuracy: **88.2%** (Sensitivity 87.5%, Specificity 88.9%)")
st.sidebar.caption("• Binary CNN Accuracy: **41.2%** (Sensitivity 12.5%, Specificity 66.7%)")

# File Upload
uploaded_file = st.file_uploader("Upload an Eye Photo (JPG, PNG, AVIF, WebP):", type=["jpg", "jpeg", "png", "bmp", "webp", "avif"])

if uploaded_file is not None:
    orig_pil = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(orig_pil)
    pil_resized = orig_pil.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_arr = np.expand_dims(np.array(pil_resized).astype(np.float32) / 255.0, axis=0)

    # 1. Feature extraction
    mean_lum, std_lum, roi_overlay, lens_mask = extract_luminance(img_rgb, IMG_SIZE)
    s1_pred = int(svm_model.predict([[mean_lum]])[0])
    s1_score = float(svm_model.decision_function([[mean_lum]])[0])
    is_cataract = (s1_pred == 1)

    # 2. Stage 2 Severity
    sev_probs = b0_severity.predict(img_arr, verbose=0)[0]
    sev_pred = int(np.argmax(sev_probs))
    sev_label = ["immature", "mature"][sev_pred]
    sev_conf = float(sev_probs[sev_pred])

    # 3. Binary CNN
    bin_probs = b0_binary.predict(img_arr, verbose=0)[0]
    bin_pred = int(np.argmax(bin_probs))
    bin_label = ["cataract", "normal"][bin_pred]
    bin_conf = float(bin_probs[bin_pred])

    # Display Columns
    col_img, col_diag = st.columns([1, 1])

    with col_img:
        st.subheader("Input Image")
        st.image(orig_pil, caption=f"{uploaded_file.name} ({orig_pil.size[0]}x{orig_pil.size[1]})", use_container_width=True)

        if show_roi:
            st.subheader("Stage 1: Lens ROI Segmentation")
            st.image(roi_overlay, caption="Extracted Lens Region (Green Overlay)", use_container_width=True)
            st.caption(f"Measured Lens Luminance: **{mean_lum:.2f}** (Normal Baseline: {stage1_params['healthy_mean_luminance']:.2f}, Diseased: {stage1_params['diseased_mean_luminance']:.2f})")

    with col_diag:
        st.subheader("Diagnostic Results")

        if eval_mode.startswith("Full Two-Stage"):
            if not is_cataract:
                st.success("### ✅ Stage 1: No Cataract Detected (Normal Eye)")
                st.metric("Detection Confidence", f"{abs(s1_score):.2f} (SVM Margin)")
                st.info("ℹ️ Lens opacification is within normal limits. Stage 2 severity evaluation skipped.")
            else:
                st.error(f"### ⚠️ Stage 1: Cataract Detected ({sev_label.title()} Stage)")
                st.metric("Severity Classification", f"{sev_label.title()} Cataract", f"{sev_conf:.1%} Confidence")

                st.write("**Severity Probabilities:**")
                st.progress(float(sev_probs[0]), text=f"Immature: {sev_probs[0]:.1%}")
                st.progress(float(sev_probs[1]), text=f"Mature: {sev_probs[1]:.1%}")

        else:
            st.write("#### Stage 1 (Luminance + SVM):")
            st.write(f"• Prediction: **{'Cataract' if is_cataract else 'Normal'}** (Score: {s1_score:.3f}, Lum: {mean_lum:.2f})")

            st.write("#### Stage 2 (EfficientNet-B0 Severity):")
            st.write(f"• Prediction: **{sev_label.title()}** ({sev_conf:.1%})")
            st.progress(float(sev_probs[0]), text=f"Immature: {sev_probs[0]:.1%}")
            st.progress(float(sev_probs[1]), text=f"Mature: {sev_probs[1]:.1%}")

            st.write("#### Direct Binary CNN (EfficientNet-B0 Binary):")
            st.write(f"• Prediction: **{bin_label.title()}** ({bin_conf:.1%})")
            st.progress(float(bin_probs[0]), text=f"Cataract: {bin_probs[0]:.1%}")
            st.progress(float(bin_probs[1]), text=f"Normal: {bin_probs[1]:.1%}")

        # Grad-CAM Attention Heatmap
        if show_cam:
            st.divider()
            st.subheader("Model Visual Attention (Grad-CAM)")
            cam_heatmap = compute_gradcam(b0_severity, img_arr, sev_pred)
            if cam_heatmap is not None:
                cam_img = overlay_gradcam(orig_pil, cam_heatmap, IMG_SIZE)
                st.image(cam_img, caption="Grad-CAM Focus Overlay (Red = Highest Activation)", use_container_width=True)

        # Save record
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image": uploaded_file.name,
            "mean_luminance": round(mean_lum, 2),
            "stage1_prediction": "cataract" if is_cataract else "normal",
            "stage2_severity": sev_label if is_cataract else None,
            "two_stage_decision": f"Cataract ({sev_label})" if is_cataract else "Normal"
        }
        with open(RESULTS_DIR / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

else:
    st.info("👆 Please upload an eye photo to evaluate.")
