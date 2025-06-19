import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import cv2
import io
import os

# Page configuration
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    
    .detection-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    
    .success-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.densenet_model = None
    st.session_state.yolo_model = None

# Model paths (you can modify these)
DENSENET_MODEL_PATH = r'C:\Users\91874\Downloads\TB\best_model_unfrozen.h5'
YOLO_MODEL_PATH = r'C:\Users\91874\Downloads\TB\best (6).pt'

# Classes
densenet_classes = ['healthy', 'sick', 'tb']
yolo_classes = ['ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']

@st.cache_resource
def load_models():
    """Load both models with caching"""
    try:
        densenet_model = tf.keras.models.load_model(DENSENET_MODEL_PATH)
        yolo_model = YOLO(YOLO_MODEL_PATH)
        return densenet_model, yolo_model, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, False

def densenet_predict(image, model):
    """DenseNet classification"""
    # Resize and preprocess image
    image_resized = image.resize((512, 512))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(image_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    return densenet_classes[predicted_idx], confidence, predictions[0]

def yolo_detect(image, model):
    """YOLO detection for TB regions"""
    # Convert PIL image to numpy array for YOLO
    img_array = np.array(image)
    
    results = model(img_array, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'class': yolo_classes[class_id]
                })
    
    return detections

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Draw rectangle
        draw.rectangle(bbox, outline='red', width=3)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        draw.text((bbox[0], bbox[1]-20), label, fill='red')
    
    return image_copy

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 TB Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Tuberculosis Detection using DenseNet Classification and YOLO Object Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a chest X-ray image
        2. Wait for the models to load
        3. View classification results
        4. If TB is detected, see region annotations
        """)
        
        st.header("⚙️ Model Information")
        st.info("""
        **DenseNet Model**: Classifies images as healthy, sick, or TB
        
        **YOLO Model**: Detects and localizes TB regions when TB is classified
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for research purposes only. 
        Always consult healthcare professionals for medical diagnosis.
        """)
    
    # Main content
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image for TB detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        
        # Load models
        if not st.session_state.models_loaded:
            with st.spinner("Loading AI models... This may take a moment."):
                densenet_model, yolo_model, success = load_models()
                if success:
                    st.session_state.densenet_model = densenet_model
                    st.session_state.yolo_model = yolo_model
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded successfully!")
                else:
                    st.error("❌ Failed to load models. Please check the model paths.")
                    return
    
    if uploaded_file is not None and st.session_state.models_loaded:
            st.header("🔍 Analysis Results")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: DenseNet Classification
            status_text.text("Step 1: Running DenseNet classification...")
            progress_bar.progress(25)
            
            classification, conf, all_predictions = densenet_predict(
                image, st.session_state.densenet_model
            )
            
            # Display classification results
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("📊 Classification Results")
            
            # Create columns for better display
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                if classification.lower() == 'tb':
                    st.markdown(f'<div class="warning-card"><strong>⚠️ Classification:</strong> {classification.upper()}</div>', unsafe_allow_html=True)
                elif classification.lower() == 'healthy':
                    st.markdown(f'<div class="success-card"><strong>✅ Classification:</strong> {classification.title()}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="detection-card"><strong>🔍 Classification:</strong> {classification.title()}</div>', unsafe_allow_html=True)
                
                st.metric("Confidence", f"{conf:.2%}")
            
            with result_col2:
                # Show all class probabilities
                st.write("**All Probabilities:**")
                for i, class_name in enumerate(densenet_classes):
                    prob = all_predictions[i]
                    st.write(f"{class_name.title()}: {prob:.2%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            progress_bar.progress(50)
            
            # Step 2: YOLO Detection (if TB detected)
            if classification.lower() == 'tb':
                status_text.text("Step 2: TB detected! Running YOLO detection...")
                progress_bar.progress(75)
                
                detections = yolo_detect(image, st.session_state.yolo_model)
                
                if detections:
                    st.subheader("🎯 TB Region Detection")
                    
                    # Create annotated image
                    annotated_image = draw_detections(image, detections)
                    
                    # Display annotated image
                    st.image(annotated_image, caption="TB Regions Detected", use_column_width=True)
                    
                    # Display detection details
                    st.write(f"**Found {len(detections)} TB region(s):**")
                    
                    for i, detection in enumerate(detections):
                        with st.expander(f"Region {i+1}: {detection['class']}"):
                            col_det1, col_det2 = st.columns(2)
                            with col_det1:
                                st.write(f"**Type:** {detection['class']}")
                                st.write(f"**Confidence:** {detection['confidence']:.2%}")
                            with col_det2:
                                bbox = detection['bbox']
                                st.write(f"**Bounding Box:**")
                                st.write(f"X1: {bbox[0]}, Y1: {bbox[1]}")
                                st.write(f"X2: {bbox[2]}, Y2: {bbox[3]}")
                    
                    # Download button for annotated image
                    buf = io.BytesIO()
                    annotated_image.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=byte_im,
                        file_name="tb_detection_result.png",
                        mime="image/png"
                    )
                else:
                    st.warning("🔍 No specific TB regions detected by YOLO model")
            else:
                status_text.text("Analysis complete - No TB detected")
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis completed!")
            
            # Summary section
            st.markdown("---")
            st.subheader("📋 Summary")
            
            summary_container = st.container()
            with summary_container:
                st.write("**Final Results:**")
                st.write(f"• **Primary Classification:** {classification.title()}")
                st.write(f"• **Confidence Level:** {conf:.2%}")
                
                if classification.lower() == 'tb' and 'detections' in locals():
                    if detections:
                        st.write(f"• **TB Regions Found:** {len(detections)}")
                        for i, det in enumerate(detections):
                            st.write(f"  - Region {i+1}: {det['class']} (Confidence: {det['confidence']:.2%})")
                    else:
                        st.write("• **TB Regions Found:** None detected")
                else:
                    st.write("• **TB Region Detection:** Not applicable")

if __name__ == "__main__":
    main()