import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from src.models.factory import ModelFactory
from src.data.preprocessor import MedicalImagePreprocessor
from src.data.mock_data import generate_synthetic_chest_xray, NIH_CLASSES

device = torch.device("cpu")
model = ModelFactory.create_model("densenet121", num_classes=14, pretrained=False)
model.to(device)
model.eval()

prep = MedicalImagePreprocessor()

test_cases = [
    ("Case 1: Normal Chest Radiograph (No Finding)", ["No Finding"]),
    ("Case 2: Cardiomegaly (Enlarged Heart)", ["Cardiomegaly"]),
    ("Case 3: Pleural Effusion (Fluid at lung base)", ["Effusion"]),
    ("Case 4: Lobar Pneumonia & Consolidation", ["Pneumonia", "Consolidation"])
]

for title, findings in test_cases:
    print("\n" + "="*65)
    print(f"DIAGNOSTIC TEST: {title}")
    print("="*65)
    img = generate_synthetic_chest_xray(size=(224, 224), findings=findings)
    tensor, _ = prep.preprocess_for_inference(img)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor))[0].cpu().numpy()

    ranked = sorted(zip(NIH_CLASSES, probs), key=lambda x: x[1], reverse=True)
    suspicious = [f"{c} ({p:.1%})" for c, p in ranked if p >= 0.35]

    for c, p in ranked:
        tag = "[!] SUSPICIOUS" if p >= 0.35 else "[Normal]"
        print(f"  {c:<20}: {p:.2%} {tag}")

    print(f"\n>> Clinical Assessment: {'SUSPICIOUS for ' + ', '.join(suspicious) if suspicious else 'NORMAL (No abnormalities detected)'}")
