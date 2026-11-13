"""ChestGuard XAI: Clinical Decision Support & Explainability Dashboard.

An interactive clinical radiograph diagnostic dashboard supporting DenseNet-121,
ResNet-50, EfficientNet, and Baseline CNNs with real-time Grad-CAM, Grad-CAM++,
and Saliency localization.
"""

import os
import io
import time
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from src.models.factory import ModelFactory
from src.data.preprocessor import MedicalImagePreprocessor
from src.xai.gradcam import GradCAM, GradCAMPlusPlus
from src.xai.saliency import VanillaSaliency
from src.xai.visualizer import create_cam_overlay, create_clinical_comparison_panel
from src.utils.report_generator import generate_clinical_html_report
from src.data.mock_data import generate_synthetic_chest_xray

# Page config
st.set_page_config(
    page_title="ChestGuard XAI - Medical Decision Support",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e88e5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #78909c;
        margin-bottom: 15px;
    }
    .disclaimer-banner {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px 15px;
        border-radius: 4px;
        color: #e65100;
        font-size: 0.88rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2e3650;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

DEFAULT_THRESHOLDS = {c: 0.25 for c in DEFAULT_CLASSES}
DEFAULT_THRESHOLDS["Pneumothorax"] = 0.20
DEFAULT_THRESHOLDS["Pneumonia"] = 0.20
DEFAULT_THRESHOLDS["Consolidation"] = 0.20
DEFAULT_THRESHOLDS["Effusion"] = 0.20
DEFAULT_THRESHOLDS["Edema"] = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_cached_model(model_name: str):
    """Loads and caches PyTorch neural network model."""
    model = ModelFactory.create_model(model_name=model_name, num_classes=14, pretrained=True)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def get_preprocessor():
    return MedicalImagePreprocessor()


# Sidebar setup
st.sidebar.title("🫁 Model & XAI Controls")

model_choice = st.sidebar.selectbox(
    "Select Architecture",
    options=["DenseNet-121 (Primary)", "ResNet-50", "EfficientNet-B0", "Baseline CNN"],
    index=0
)
model_key = {
    "DenseNet-121 (Primary)": "densenet121",
    "ResNet-50": "resnet50",
    "EfficientNet-B0": "efficientnet_b0",
    "Baseline CNN": "baseline_cnn"
}[model_choice]

xai_method = st.sidebar.selectbox(
    "Explainability Algorithm",
    options=["Grad-CAM", "Grad-CAM++", "Pixel Saliency Map"],
    index=0
)

colormap = st.sidebar.selectbox(
    "Heatmap Colormap",
    options=["jet", "viridis", "inferno", "turbo"],
    index=0
)

cam_alpha = st.sidebar.slider(
    "Heatmap Blend Opacity (α)",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick-Load Clinical Presets")
preset_choice = st.sidebar.selectbox(
    "Load Clinical Case",
    options=[
        "Upload Custom Image",
        "Case 1: Cardiomegaly (Enlarged Heart)",
        "Case 2: Pleural Effusion (Fluid in Pleura)",
        "Case 3: Lobar Pneumonia (Alveolar Opacity)",
        "Case 4: Solitary Pulmonary Nodule",
        "Case 5: Normal Chest Radiograph"
    ]
)

# Main Banner
st.markdown('<div class="main-header">Explainable Medical Image Classification System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Label Chest Radiograph Pathology Detection with Grad-CAM Decision-Support Heatmaps</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <b>RESEARCH & EDUCATIONAL PROTOTYPE ONLY:</b> This application provides explainable artificial intelligence (XAI)
    decision support. Heatmaps and probabilities are investigational and must NOT be used as a primary diagnostic tool.
</div>
""", unsafe_allow_html=True)

# Image acquisition
loaded_image = None
case_findings = []

if preset_choice == "Upload Custom Image":
    uploaded_file = st.file_uploader(
        "Upload Chest Radiograph (PNG, JPG, DICOM-converted)",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        loaded_image = Image.open(uploaded_file)
else:
    preset_map = {
        "Case 1: Cardiomegaly (Enlarged Heart)": ["Cardiomegaly"],
        "Case 2: Pleural Effusion (Fluid in Pleura)": ["Effusion"],
        "Case 3: Lobar Pneumonia (Alveolar Opacity)": ["Pneumonia", "Consolidation"],
        "Case 4: Solitary Pulmonary Nodule": ["Nodule"],
        "Case 5: Normal Chest Radiograph": ["No Finding"]
    }
    case_findings = preset_map[preset_choice]
    loaded_image = generate_synthetic_chest_xray(size=(224, 224), findings=case_findings)

if loaded_image is None:
    st.info("👈 Please select a clinical case preset from the sidebar or upload a chest X-ray image to begin analysis.")
else:
    preprocessor = get_preprocessor()
    model = load_cached_model(model_key)

    tensor, display_img = preprocessor.preprocess_for_inference(loaded_image)
    tensor = tensor.to(DEVICE)

    # Model inference
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Predictions dict
    pred_dict = {DEFAULT_CLASSES[i]: float(probs[i]) for i in range(len(DEFAULT_CLASSES))}
    sorted_findings = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    top_pathology, top_prob = sorted_findings[0]

    # Metrics summary cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Model", model_choice.split()[0], f"{latency_ms:.1f} ms")
    m2.metric("Primary Suspicion", top_pathology, f"{top_prob:.1%}")
    pos_count = sum(1 for c, p in pred_dict.items() if p >= DEFAULT_THRESHOLDS.get(c, 0.5))
    m3.metric("Abnormal Findings", f"{pos_count} of 14", "Clinical Flags")
    m4.metric("Inference Engine", str(DEVICE).upper(), "FP32")

    st.markdown("---")

    # Layout: Diagnostic Visualizer & Findings
    col_vis, col_pred = st.columns([1.1, 1.0])

    with col_vis:
        st.subheader("Visual Explainability & Localization")

        # Pathology focus selector for XAI
        selected_cam_disease = st.selectbox(
            "Highlight Anatomical Region for Pathology:",
            options=DEFAULT_CLASSES,
            index=DEFAULT_CLASSES.index(top_pathology)
        )
        cam_idx = DEFAULT_CLASSES.index(selected_cam_disease)

        target_layer = ModelFactory.get_gradcam_target_layer(model, model_key)

        if xai_method == "Grad-CAM++":
            explainer = GradCAMPlusPlus(model, target_layer)
            heatmap = explainer.generate_heatmap(tensor, cam_idx)
        elif xai_method == "Pixel Saliency Map":
            explainer = VanillaSaliency(model)
            heatmap = explainer.generate_saliency(tensor, cam_idx)
        else:
            explainer = GradCAM(model, target_layer)
            heatmap = explainer.generate_heatmap(tensor, cam_idx)

        overlay = create_cam_overlay(display_img, heatmap, alpha=cam_alpha, colormap=colormap)

        vis_tabs = st.tabs(["Overlay Heatmap", "Side-by-Side Comparison", "Raw Attributions"])
        with vis_tabs[0]:
            st.image(
                overlay,
                caption=f"{xai_method} Activation Map for '{selected_cam_disease}' (P = {pred_dict[selected_cam_disease]:.1%})",
                use_container_width=True
            )
        with vis_tabs[1]:
            panel = create_clinical_comparison_panel(
                original_img=display_img,
                gradcam_overlay=create_cam_overlay(display_img, GradCAM(model, target_layer).generate_heatmap(tensor, cam_idx), alpha=cam_alpha),
                gradcam_pp_overlay=create_cam_overlay(display_img, GradCAMPlusPlus(model, target_layer).generate_heatmap(tensor, cam_idx), alpha=cam_alpha),
                saliency_map=VanillaSaliency(model).generate_saliency(tensor, cam_idx),
                pathology_name=selected_cam_disease,
                probability=pred_dict[selected_cam_disease],
                threshold=DEFAULT_THRESHOLDS.get(selected_cam_disease, 0.5)
            )
            st.image(panel, use_container_width=True)
        with vis_tabs[2]:
            st.image(heatmap, caption="Normalized Spatial Heatmap Matrix [0, 1]", use_container_width=True, clamp=True)

    with col_pred:
        st.subheader("14 Multi-Label Pathology Predictions")

        # Interactive probability bars
        chart_data = []
        for disease, prob in sorted_findings:
            thresh = DEFAULT_THRESHOLDS.get(disease, 0.5)
            is_pos = prob >= thresh
            risk = "High" if prob >= 0.70 else ("Moderate" if is_pos else "Low")
            chart_data.append({
                "Pathology": disease,
                "Probability": prob,
                "Threshold": thresh,
                "Status": "SUSPICIOUS" if is_pos else "Normal",
                "Risk": risk
            })

        df_chart = pd.DataFrame(chart_data)
        st.dataframe(
            df_chart.style.format({"Probability": "{:.1%}", "Threshold": "{:.2f}"})
            .background_gradient(subset=["Probability"], cmap="YlOrRd"),
            use_container_width=True,
            height=380
        )

    st.markdown("---")
    st.subheader("Downloadable Clinical Decision Support Summary")

    # Report Generator
    patient_meta = {
        "patient_id": "PT_DEMO_4821",
        "age": "62",
        "gender": "M",
        "view_position": "PA"
    }

    report_html = generate_clinical_html_report(
        patient_info=patient_meta,
        predictions=pred_dict,
        thresholds=DEFAULT_THRESHOLDS,
        original_img=display_img,
        gradcam_img=overlay,
        model_name=model_choice
    )

    st.download_button(
        label="📄 Download Comprehensive Clinical Summary (HTML)",
        data=report_html,
        file_name=f"chest_xray_report_{model_key}_{int(time.time())}.html",
        mime="text/html"
    )
