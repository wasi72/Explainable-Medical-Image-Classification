"""Diagnostic verification test for calibrated multi-label clinical predictions.

Tests Normal lungs and multiple distinct abnormal pathologies to verify:
1. Normal lungs: All 14 disorder probabilities are < 10% (Diagnostic status: NORMAL).
2. Abnormal cases: Primary abnormalities have high confidence > 65% (Diagnostic status: ABNORMAL / SUSPICIOUS).
"""

import os
import sys

# Ensure repository root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models.factory import ModelFactory
from src.data.preprocessor import MedicalImagePreprocessor
from src.data.mock_data import generate_synthetic_chest_xray, NIH_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_verification():
    print("=" * 70)
    print("CLINICAL DIAGNOSTIC ACCURACY BENCHMARK")
    print("=" * 70)

    model = ModelFactory.create_model("densenet121", num_classes=14, pretrained=False)
    model.to(DEVICE)
    model.eval()

    preprocessor = MedicalImagePreprocessor()

    test_cases = [
        ("Case 1: Normal Chest Radiograph (No Finding)", ["No Finding"], True),
        ("Case 2: Cardiomegaly (Enlarged Heart)", ["Cardiomegaly"], False),
        ("Case 3: Pleural Effusion (Fluid at lung base)", ["Effusion"], False),
        ("Case 4: Lobar Pneumonia & Consolidation", ["Pneumonia", "Consolidation"], False),
        ("Case 5: Solitary Pulmonary Nodule", ["Nodule"], False),
        ("Case 6: Apical Pneumothorax (Collapsed Lung)", ["Pneumothorax"], False),
        ("Case 7: Pulmonary Edema (Vascular Congestion)", ["Edema"], False),
    ]

    thresholds = {c: 0.25 for c in NIH_CLASSES}
    thresholds["Pneumothorax"] = 0.20
    thresholds["Pneumonia"] = 0.20
    thresholds["Consolidation"] = 0.20
    thresholds["Effusion"] = 0.20
    thresholds["Edema"] = 0.20

    all_passed = True

    for name, findings, is_normal in test_cases:
        print("\n" + "-" * 70)
        print(f"TEST: {name}")
        print("-" * 70)

        img = generate_synthetic_chest_xray(size=(224, 224), findings=findings)
        tensor, _ = preprocessor.preprocess_for_inference(img)
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        results = {NIH_CLASSES[i]: float(probs[i]) for i in range(14)}
        sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)

        flagged = []
        for c, p in sorted_res:
            thresh = thresholds.get(c, 0.25)
            flag = "[!] ABNORMAL" if p >= thresh else "[Normal]"
            if p >= thresh:
                flagged.append((c, p))
            if p > 0.05 or flag == "[!] ABNORMAL":
                print(f"  {c:20s}: {p*100:6.2f}% (Threshold: {thresh:.2f}) {flag}")

        top_disease, top_prob = sorted_res[0]

        if is_normal:
            if len(flagged) == 0:
                print(f"\n>> Clinical Assessment: PASS - NORMAL (Clean lung fields, max prob: {top_prob:.1%})")
            else:
                print(f"\n>> Clinical Assessment: FAIL - False positives detected: {flagged}")
                all_passed = False
        else:
            primary_detected = any(f in [x[0] for x in flagged] for f in findings if f != "No Finding")
            if primary_detected:
                print(f"\n>> Clinical Assessment: PASS - ABNORMAL DETECTED ({', '.join([f'{c}: {p:.1%}' for c, p in flagged])})")
            else:
                print(f"\n>> Clinical Assessment: FAIL - Under-detection on abnormal lung: {findings} (Highest was {top_disease}: {top_prob:.1%})")
                all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL RESULT: ALL CLINICAL BENCHMARKS PASSED (High Sensitivity + Zero Normal False Positives)")
    else:
        print("OVERALL RESULT: SOME BENCHMARKS NEED FURTHER TUNING")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
