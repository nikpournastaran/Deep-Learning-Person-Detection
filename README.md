# Deep-Learning-Person-Detection
# 🚶‍♂️ Real-time Pedestrian Detection using YOLOv8

![Detection Example](https://img.shields.io/badge/Computer_Vision-Object_Detection-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)

## 📌 Project Overview
This repository contains a high-performance deep learning pipeline designed to detect pedestrians in various environments. Leveraging the **YOLOv8 (You Only Look Once)** architecture, the model is optimized for both accuracy and real-time inference speed.

## 🛠 Technical Stack
* **Core Framework:** PyTorch
* **Model Architecture:** Ultralytics YOLOv8 (Nano version for efficiency)
* **Data Processing:** NumPy, Pandas, XML ETree
* **Visualization:** OpenCV, Matplotlib, Seaborn
* **Environment:** Kaggle GPU (NVIDIA Tesla T4)

## 🚀 Key Features
* **Automated Label Conversion:** Custom scripts to transform XML annotations into normalized YOLO format $(x, y, w, h)$.
* **Optimized Training:** Utilizing pre-trained weights for transfer learning to reduce training time and improve convergence.
* **Robust Evaluation:** Performance metrics analyzed using precision-recall curves and loss visualizations.

## 📁 Dataset Structure
The model is trained on a dedicated Pedestrian Detection dataset, categorized into:
- `Person`: Standard pedestrian instances.
- `Person-like`: Objects frequently misidentified as humans (to reduce False Positives).

## 💻 Quick Start
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nikpournastaran/Deep-Learning-Person-Detection.git](https://github.com/nikpournastaran/Deep-Learning-Person-Detection.git)
   cd Deep-Learning-Person-Detection

2. **Install dependencies:**
```bash
pip install ultralytics opencv-python matplotlib


3. **Inference Example:**
   from ultralytics import YOLO

# Load the trained weights
model = YOLO('yolov8n.pt') 

# Run detection
results = model.predict(source='path/to/your/image.jpg', save=True)



