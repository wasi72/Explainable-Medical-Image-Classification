"""CLI script to run inference and generate Grad-CAM heatmaps for a single chest radiograph."""

import os
import sys

# Ensure repository root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from PIL import Image
import torch

from src.models.factory import ModelFactory
from src.data.preprocessor import MedicalImagePreprocessor
from src.xai.gradcam import GradCAM, GradCAMPlusPlus
from src.xai.saliency import VanillaSaliency
from src.xai.visualizer import create_cam_overlay
from src.data.mock_data import generate_synthetic_chest_xray, NIH_CLASSES
from src.utils.logger import get_logger

logger = get_logger("InferenceCLI")


def main():
    parser = argparse.ArgumentParser(description="Single-image radiograph inference & Grad-CAM")
    parser.add_argument("--image", type=str, default=None, help="Path to input radiograph image")
    parser.add_argument("--model", type=str, default="densenet121", choices=["densenet121", "resnet50", "efficientnet_b0", "baseline_cnn"])
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "gradcam_plus_plus", "saliency"])
    parser.add_argument("--pathology", type=str, default=None, help="Specific pathology to explain")
    parser.add_argument("--output", type=str, default="reports/inference_overlay.png")
    parser.add_argument("--colormap", type=str, default="jet")
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    preprocessor = MedicalImagePreprocessor()

    if args.image is None or not os.path.exists(args.image):
        logger.info("No input image specified or file not found. Generating sample Cardiomegaly test image...")
        os.makedirs("data/sample_images", exist_ok=True)
        sample_path = "data/sample_images/demo_cardiomegaly.png"
        img = generate_synthetic_chest_xray(size=(224, 224), findings=["Cardiomegaly"])
        img.save(sample_path)
        input_image = sample_path
    else:
        input_image = args.image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model '{args.model}' on {device}...")
    model = ModelFactory.create_model(args.model, num_classes=14, pretrained=True).to(device)
    model.eval()

    tensor, display_img = preprocessor.preprocess_for_inference(input_image)
    tensor = tensor.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    print("\n" + "="*50)
    print("      MULTI-LABEL PREDICTION RESULTS")
    print("="*50)
    ranked = sorted(zip(NIH_CLASSES, probs), key=lambda x: x[1], reverse=True)
    for disease, prob in ranked:
        flag = " [!] SUSPICIOUS" if prob >= 0.50 else ""
        print(f" - {disease:<20}: {prob:.1%}{flag}")
    print("="*50 + "\n")

    # Select target pathology for XAI
    if args.pathology and args.pathology in NIH_CLASSES:
        target_idx = NIH_CLASSES.index(args.pathology)
        target_name = args.pathology
    else:
        target_idx = int(np.argmax(probs))
        target_name = NIH_CLASSES[target_idx]

    logger.info(f"Generating {args.method} for target pathology: '{target_name}' (P = {probs[target_idx]:.1%})")
    target_layer = ModelFactory.get_gradcam_target_layer(model, args.model)

    if args.method == "gradcam_plus_plus":
        explainer = GradCAMPlusPlus(model, target_layer)
        cam = explainer.generate_heatmap(tensor, target_idx)
    elif args.method == "saliency":
        explainer = VanillaSaliency(model)
        cam = explainer.generate_saliency(tensor, target_idx)
    else:
        explainer = GradCAM(model, target_layer)
        cam = explainer.generate_heatmap(tensor, target_idx)

    overlay = create_cam_overlay(display_img, cam, alpha=args.alpha, colormap=args.colormap)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    logger.info(f"Saved Grad-CAM overlay visualization to: {args.output}")


if __name__ == "__main__":
    main()
