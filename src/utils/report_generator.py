"""Clinical Decision Support Report Generator.

Synthesizes model predictions, explainability heatmaps, patient demographics,
and regulatory disclaimers into clean printable HTML clinical summaries.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import base64
import numpy as np
from PIL import Image

from src.xai.visualizer import image_to_base64


def generate_clinical_html_report(
    patient_info: Dict[str, Any],
    predictions: Dict[str, float],
    thresholds: Dict[str, float],
    original_img: np.ndarray,
    gradcam_img: np.ndarray,
    model_name: str = "DenseNet-121",
    output_path: Optional[str] = None
) -> str:
    """Generates an HTML clinical decision support summary."""
    patient_id = patient_info.get("patient_id", "ANON_PT_001")
    age = patient_info.get("age", "N/A")
    gender = patient_info.get("gender", "N/A")
    view = patient_info.get("view_position", "PA")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine findings
    positive_findings = []
    for disease, prob in predictions.items():
        thresh = thresholds.get(disease, 0.5)
        if prob >= thresh:
            positive_findings.append((disease, prob, thresh))

    # Sort by probability descending
    positive_findings.sort(key=lambda x: x[1], reverse=True)

    orig_b64 = image_to_base64(original_img)
    cam_b64 = image_to_base64(gradcam_img)

    # Build pathology rows
    rows_html = ""
    for disease, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        thresh = thresholds.get(disease, 0.5)
        is_pos = prob >= thresh
        status_badge = (
            f'<span style="background-color: #ffebee; color: #c62828; padding: 4px 8px; border-radius: 4px; font-weight: bold;">SUSPICIOUS ({prob:.1%})</span>'
            if is_pos else
            f'<span style="background-color: #e8f5e9; color: #2e7d32; padding: 4px 8px; border-radius: 4px;">Low Risk ({prob:.1%})</span>'
        )
        bar_color = "#e53935" if is_pos else "#43a047"
        pct = min(100, int(prob * 100))

        rows_html += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #e0e0e0;"><b>{disease}</b></td>
            <td style="padding: 10px; border-bottom: 1px solid #e0e0e0;">
                <div style="background-color: #eee; border-radius: 4px; width: 100%; height: 16px;">
                    <div style="background-color: {bar_color}; width: {pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </td>
            <td style="padding: 10px; border-bottom: 1px solid #e0e0e0; text-align: center;">{thresh:.2f}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e0e0e0; text-align: center;">{status_badge}</td>
        </tr>
        """

    impression = (
        f"Automated analysis identified suspicious radiological findings for: "
        f"<b>{', '.join([f[0] for f in positive_findings])}</b>."
        if positive_findings else
        "No cardiopulmonary abnormalities exceed clinical operating threshold."
    )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Clinical Radiograph AI Decision Support Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; }}
        .header {{ border-bottom: 3px solid #1976d2; padding-bottom: 15px; margin-bottom: 25px; }}
        .title {{ font-size: 24px; font-weight: bold; color: #0d47a1; }}
        .meta-table {{ width: 100%; margin-bottom: 25px; border-collapse: collapse; }}
        .meta-table td {{ padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; }}
        .disclaimer {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin-top: 30px; font-size: 13px; color: #856404; }}
        .images-container {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .img-box {{ flex: 1; text-align: center; border: 1px solid #ddd; padding: 10px; border-radius: 6px; background: #fafafa; }}
        .img-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .section-title {{ font-size: 18px; font-weight: bold; margin: 20px 0 10px 0; color: #1565c0; }}
        table.results {{ width: 100%; border-collapse: collapse; }}
        table.results th {{ background-color: #1976d2; color: white; padding: 10px; text-align: left; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">CHEST RADIOGRAPH AI DECISION SUPPORT REPORT</div>
        <div style="color: #666; font-size: 13px; margin-top: 5px;">Powered by Explainable Deep Learning (DenseNet-121 & Grad-CAM)</div>
    </div>

    <table class="meta-table">
        <tr>
            <td><b>Patient ID:</b> {patient_id}</td>
            <td><b>Age:</b> {age}</td>
            <td><b>Gender:</b> {gender}</td>
            <td><b>View Position:</b> {view}</td>
        </tr>
        <tr>
            <td><b>Analysis Date:</b> {timestamp}</td>
            <td><b>AI Model:</b> {model_name}</td>
            <td><b>Explainability:</b> Grad-CAM (Target: Conv4/DenseBlock4)</td>
            <td><b>Status:</b> Completed</td>
        </tr>
    </table>

    <div class="section-title">Visual Evidence & Localization</div>
    <div class="images-container">
        <div class="img-box">
            <h4>Original Chest Radiograph</h4>
            <img src="{orig_b64}" alt="Original X-ray">
        </div>
        <div class="img-box">
            <h4>Explainable AI (Grad-CAM Heatmap)</h4>
            <img src="{cam_b64}" alt="Grad-CAM Overlay">
        </div>
    </div>

    <div class="section-title">Automated AI Impression</div>
    <div style="background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; font-size: 15px; margin-bottom: 25px;">
        {impression}
    </div>

    <div class="section-title">Pathology Detection Probabilities (14 Multi-Label Classes)</div>
    <table class="results">
        <thead>
            <tr>
                <th>Pathology</th>
                <th style="width: 35%;">Probability Bar</th>
                <th style="text-align: center;">Cutoff Threshold</th>
                <th style="text-align: center;">Clinical Classification</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <div class="disclaimer">
        <b>REGULATORY & MEDICAL DISCLAIMER:</b><br>
        This analysis is generated by an experimental research AI prototype for investigational decision-support purposes only.
        It is NOT an FDA-cleared diagnostic device and should never replace evaluation by a certified radiologist or physician.
        Treatment, prescription, and triage decisions must never be based solely on this automated output.
    </div>
</body>
</html>
"""

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
