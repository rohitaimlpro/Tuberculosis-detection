# 🔬 TB Detection System

An AI-powered web application for detecting **Tuberculosis (TB)** from **chest X-ray images** using **DenseNet for image classification** and **YOLO for object detection**. The application is built with **Streamlit**, providing a simple and interactive interface to upload images and view detection results with confidence scores and annotated bounding boxes.

---

## 🚀 Features

- ✅ Classifies images into **Healthy**, **Sick**, or **TB**
- ✅ Detects and localizes TB-affected regions in X-rays
- ✅ Interactive Streamlit interface
- ✅ Confidence probabilities for all classes
- ✅ Option to download annotated detection images

---

## 🧠 Models Overview

| Model     | Task              | Classes Detected |
|-----------|-------------------|------------------|
| DenseNet  | Classification    | healthy, sick, tb |
| YOLOv8    | Object Detection  | ActiveTuberculosis, ObsoletePulmonaryTuberculosis |

### 📊 Dataset

The models were trained on the **TBX11K dataset**, a large-scale tuberculosis X-ray dataset containing:
- **11,200 chest X-ray images**
- Multiple TB manifestations including active and obsolete pulmonary tuberculosis
- Expert annotations for both classification and localization tasks

**Dataset Reference:** [TBX11K: A Large-scale Tuberculosis X-ray Dataset](https://mmcheng.net/tb/)

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        A[Streamlit Web Interface] --> B[Image Upload Component]
        B --> C[Results Display]
    end
    
    subgraph "Processing Pipeline"
        D[Image Preprocessing] --> E[DenseNet Classification]
        E --> F{TB Detected?}
        F -->|Yes| G[YOLO Object Detection]
        F -->|No| H[Display Classification Only]
        G --> I[Bounding Box Annotation]
    end
    
    subgraph "AI Models"
        J[DenseNet Model<br/>best_model_unfrozen.h5]
        K[YOLOv8 Model<br/>best.pt]
    end
    
    subgraph "Output"
        L[Classification Results<br/>- Class: healthy/sick/tb<br/>- Confidence Score<br/>- All Probabilities]
        M[Detection Results<br/>- TB Region Localization<br/>- Annotated X-ray Image<br/>- Detection Confidence]
    end
    
    B --> D
    E -.loads.-> J
    G -.loads.-> K
    H --> L
    I --> M
    L --> C
    M --> C
    
    style A fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    style J fill:#50C878,stroke:#2E7D4E,stroke-width:2px,color:#fff
    style K fill:#50C878,stroke:#2E7D4E,stroke-width:2px,color:#fff
    style F fill:#FFB84D,stroke:#CC8A3D,stroke-width:2px,color:#000
    style L fill:#9B59B6,stroke:#6C3A7C,stroke-width:2px,color:#fff
    style M fill:#9B59B6,stroke:#6C3A7C,stroke-width:2px,color:#fff
```

### Architecture Components:

1. **User Interface Layer**
   - Built with Streamlit for interactive web experience
   - Drag-and-drop image upload functionality
   - Real-time results visualization

2. **Processing Pipeline**
   - Image preprocessing (resizing, normalization)
   - Sequential processing: Classification → Detection
   - Conditional execution based on TB detection

3. **AI Models**
   - **DenseNet**: Deep learning classifier for 3-class categorization
   - **YOLOv8**: Real-time object detector for TB region localization

4. **Output Layer**
   - Classification probabilities for all classes
   - Annotated images with bounding boxes
   - Downloadable results

---

## 📁 Project Structure

```
TB-Detection-System/
│
├── app.py                      # Main application file
├── best_model_unfrozen.h5      # DenseNet model (Classification)
├── best (6).pt                 # YOLO model (Detection)
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── images/                     # Application screenshots
    ├── Screenshot 2025-10-25 154235.png
    ├── Screenshot 2025-10-25 154458.png
    ├── Screenshot 2025-10-25 154524.png
    └── Screenshot 2025-10-25 154613.png
```

---

## 🔧 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/rohitaimlpro/Tuberculosis-detection.git
cd Tuberculosis-detection
```

### 2️⃣ Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Model Files

Edit these lines in `app.py` to your actual model paths:

```python
DENSENET_MODEL_PATH = 'path/to/best_model_unfrozen.h5'
YOLO_MODEL_PATH = 'path/to/best (6).pt'
```

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📸 How to Use

1. Launch the app
2. Upload a chest X-ray image (`.png`, `.jpg`, or `.jpeg`)
3. The system will:
   - Classify the image using DenseNet
   - If TB is detected, YOLO will highlight TB regions
4. View the results with confidence scores
5. Download the annotated image (if available)

---

## 📷 Application Screenshots

### 1. Home Screen - Upload Interface
![Home Screen](screenshots/home_screen.png)
*Upload your chest X-ray image using drag-and-drop or browse files*

### 2. Image Upload & Processing
![Image Upload](screenshots/image_upload.png)
*The application processes the uploaded X-ray image*

### 3. Classification Results
![Classification Results](screenshots/classification_results.png)
*View detailed classification results with confidence scores for all classes*

### 4. TB Region Detection
![TB Detection](screenshots/tb_detection.png)
*YOLO model highlights and annotates TB-affected regions in the X-ray*

---

## ✅ Example Output

**Classification Result:**
```
Classification: TB
Confidence: 92.34%
```

**Detected Regions:**
```
Region 1: ActiveTuberculosis (Confidence: 88.12%)
```

---

## 📦 Requirements

Create a `requirements.txt` file with the following:

```
streamlit==1.28.0
tensorflow==2.15.0
ultralytics==8.0.200
opencv-python==4.8.1.78
pillow==10.1.0
numpy==1.24.3
```

**Note:** Version numbers may vary based on your Python version and system requirements.

---

## ⚠️ Disclaimer

> **This application is intended only for research and educational purposes.**  
> It should **not** be used as a substitute for professional medical diagnosis.

---

## 🌟 Future Enhancements

- ☁️ Cloud deployment (AWS/Azure/GCP)
- 🔍 Grad-CAM visualization for interpretability
- 📊 Detailed medical reporting
- 🩺 Multi-modal health analytics
- 📱 Mobile application support
- 🔐 Patient data management with HIPAA compliance

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

- **Author:** Rohit Sharma
- **GitHub:** [https://github.com/rohitaimlpro](https://github.com/rohitaimlpro)
- **Email:** rs5294645@gmail.com

---

⭐ **If you found this project useful, please give it a star!**