# Complete ML Project Plan: Explainable Medical Image Classification Using Deep Learning

(Recommended portfolio + research project) (Intermediate to Advanced)

I recommend building this as a complete AI research and engineering project, not just a model that classifies medical images.

Your project will combine\:

1. Medical image processing.
2. Deep Learning and Computer Vision.
3. Multi-label disease classification.
4. Explainable Artificial Intelligence (XAI).
5. Model performance evaluation.
6. Model testing and validation.
7. Web application development.
8. Research documentation and GitHub deployment.

The goal is to build something that looks like a professional AI research project that you can present to employers, universities and potential PhD supervisors.

# 1\. Project overview



# Explainable AI-Based Medical Image Classification System

**Full project title\:**

*An Explainable Deep Learning Framework for Multi-Label Chest X-Ray Disease Classification Using CNNs and Grad-CAM.*

**Main objective\:**

Develop an AI system that analyzes chest X-ray images, predicts possible abnormalities and visually highlights image regions that influence its predictions.

### What will your system do?

Imagine a user uploads a chest X-ray image.

Your system will\:

1. Accept the medical image.
2. Preprocess and resize it.
3. Pass it through a trained deep learning model.
4. Predict the probability of different abnormalities.
5. Generate an explainability heatmap.
6. Display the predictions and heatmap.
7. Generate a downloadable analysis report.



**Important\:** This is a research and educational decision-support prototype, not a medical diagnostic product. Predictions and heatmaps must not be used to make treatment decisions.

# 2\. Select your dataset

The dataset is one of the most important decisions in this project.

I recommend starting with NIH ChestX-ray14.

**NIH ChestX-ray14** (Recommended)



| Dataset information | Details |
|--------|--------|
| Images | 112,120 |
| Patients | Approximately 30,800 |
| Image type | Chest X-ray |
| Disease categories | 14 |
| Format | PNG |
| Task | Multi-label classification |
| Source | NIH Clinical Center |

Dataset characteristics: 

[Access NIH dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)

### The 14 disease categories

Your model should learn to identify these abnormalities\:

| No. | Disease |
|--------|--------|
| 1 | Atelectasis |
| 2 | Cardiomegaly |
| 3 | Effusion |
| 4 | Infiltration |
| 5 | Mass |
| 6 | Nodule |
| 7 | Pneumonia |
| 8 | Pneumothorax |
| 9 | Consolidation |
| 10 | Edema |
| 11 | Emphysema |
| 12 | Fibrosis |
| 13 | Pleural Thickening |
| 14 | Hernia |

One image can have multiple abnormalities. Therefore, this is a **multi-label classification problem**, not ordinary single-class classification.

### Other datasets for future improvements

| Dataset | Purpose |
|--------|--------|
|  | Additional chest X-ray evaluation |
|  | External validation |
|  | Expert-annotated findings |

MIMIC-CXR requires credentialed access, so it is better as a later research extension. 

# 3\. System architecture

Your project should have a clearly defined architecture.

## Complete AI pipeline

**Input Chest X-ray Image**

PNG / JPG

**Image Preprocessing**

Resize • Normalize • Quality checks

**Deep Learning Model**

DenseNet121 / ResNet50 / EfficientNet

**14 Disease Probabilities**

Multi-label prediction

**Explainable AI**

Grad-CAM heatmaps

**Evaluation**

Metrics and confidence analysis

**Interactive Web Dashboard**

Predictions • Heatmaps • Reports

# 4\. Technologies and tools

Use the following technology stack.

| Component | Technology |
|--------|--------|
| Programming | Python |
| Data processing | NumPy, Pandas |
| Image processing | OpenCV, Pillow |
| Deep learning | PyTorch |
| Pretrained models | Torchvision |
| Explainability | Grad-CAM, Captum |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Experiment tracking | MLflow |
| API | FastAPI |
| Frontend | Streamlit |
| Testing | Pytest |
| Version control | Git and GitHub |
| Deployment | Docker |

**My recommendation\:** Use PyTorch instead of trying to learn multiple deep learning frameworks simultaneously. It will make your research experiments easier to organize.

# 5\. Machine Learning methodology

This is the most important part of the project.

You should not train only one model. Build multiple models and compare them scientifically.

### Model 1: CNN baseline

Start with a simple convolutional neural network.

Purpose\:

- Understand image classification.
- Learn convolution and pooling.
- Establish a baseline.

### Model 2: ResNet50


Use pretrained ResNet50 with transfer learning.

Purpose\:

- Learn residual connections.
- Compare feature extraction.
- Establish a stronger baseline.

### Model 3: DenseNet121


(Primary model recommendation)

DenseNet121 is a strong starting point for chest X-ray classification research.

Purpose\:

- Transfer learning.
- Feature reuse.
- Multi-label prediction.
- Grad-CAM explanations.

### Model 4: EfficientNet-B0

Use EfficientNet as another comparison model.

Purpose\:

- Evaluate computational efficiency.
- Compare performance against DenseNet121.
- Measure inference time.

### Model comparison

| Model | Role |
|--------|--------|
| Custom CNN | Baseline |
| ResNet50 | Comparison |
| DenseNet121 | Main model |
| EfficientNet-B0 | Efficiency comparison |

Do not assume DenseNet121 will win. Your experiments must determine which performs best.

---

# 6\. Image preprocessing

Before training, prepare the images properly.

Your preprocessing pipeline should contain\:

**Step 1: Load image**

Read PNG images using Pillow or OpenCV.

**Step 2: Check image quality**

- Check missing or corrupted images.
- Verify image dimensions.
- Check pixel values.
- Identify duplicates.

**Step 3: Resize**

Use 224 × 224 pixels as your initial image size.

**Step 4: Convert grayscale to RGB**

Pretrained ImageNet models generally expect three-channel inputs. Replicate the grayscale channel three times.

**Step 5: Normalize**

For pretrained ImageNet models, begin with the ImageNet mean and standard deviation.

**Step 6: Data augmentation**

Apply training-only transformations\:

- Small rotations.
- Mild translations.
- Limited contrast adjustments.
- Small affine transformations.

Avoid aggressive transformations that could remove or alter clinically meaningful features.

**Important\:** Do not randomly split individual images if images from the same patient can appear multiple times. Split by patient ID to prevent data leakage.

A good initial experimental split is\:

**Patient-level data splitting**

# 70%

**Training**

# 10%

**Validation**

# 20%

**Testing**

Illustrative target split. Preserve the official NIH train/test separation and create the validation set only from the training patients.

# 7\. Model training strategy

### How the model learns

The model receives an image and predicts 14 probabilities.

For example, the output might look like this\:

| Disease | Example probability |
|--------|--------|
| Atelectasis | 0.72 |
| Cardiomegaly | 0.15 |
| Effusion | 0.83 |
| Pneumonia | 0.64 |
| Pneumothorax | 0.08 |

These are illustrative model outputs, not actual clinical results.

### Loss function

Use Binary Cross Entropy with Logits\:

\[
L=-\frac{1}{N}\sum_{i=1}^{N}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
\]

In PyTorch, use\:



```python
torch.nn.BCEWithLogitsLoss()
```



For class imbalance, investigate class-specific positive weights or focal loss. Do not automatically apply multiple balancing techniques together; compare their effects experimentally.

### Initial training configuration

| Parameter | Starting value |
|--------|--------|
| Architecture | DenseNet121 |
| Image size | 224 × 224 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Epochs | 20 |
| Loss | BCEWithLogitsLoss |
| Scheduler | ReduceLROnPlateau |
| Early stopping | 4–5 epochs |
| Random seed | 42 |

These are starting hyperparameters, not guaranteed optimal values. Tune them using validation data.

### Training procedure

1. Load pretrained DenseNet121.
2. Replace its final classification layer with 14 outputs.
3. Freeze the feature extractor initially.
4. Train the classification head.
5. Unfreeze selected layers.
6. Fine-tune with a smaller learning rate.
7. Save the best validation checkpoint.
8. Evaluate once on the held-out test set.

# 17\. My recommendation: Build it in three stages

Do not try to build everything simultaneously.



**Stage 1: Working ML model**

Dataset, preprocessing, DenseNet121, training and evaluation.

(First priority)

---



**Stage 2: Research contribution**

Grad-CAM, Integrated Gradients, model comparisons and robustness testing.

---



**Stage 3: Professional product**

Streamlit, FastAPI, Docker, automated testing and GitHub documentation.

My advice is to make DenseNet121 your first working model, then add the explainability and research experiments. This will keep the project manageable.

**Your immediate next step\:** Set up the Python environment, download NIH ChestX-ray14 and create the GitHub folder structure. Don't start training until your dataset and patient-level splits are verified.

I can also review your progress every Sunday and give you the next week's implementation tasks.

Weekly ML project review