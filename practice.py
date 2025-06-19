# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load model
# model = tf.keras.models.load_model(r'C:\Users\91874\Downloads\TB\best_model_unfrozen.h5')

# # Classes
# classes = ['health', 'sick', 'tb']

# # Inference function
# def predict(image_path):
#     # Load and preprocess image
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize((512, 512))
#     image_array = np.array(image) / 255.0  # Normalize to [0,1]
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
#     # Make prediction
#     predictions = model.predict(image_array)
#     predicted_idx = np.argmax(predictions[0])
#     confidence = predictions[0][predicted_idx]
    
#     return classes[predicted_idx], confidence

# # Usage
# result, confidence = predict('h100.png')
# print(f"Prediction: {result}, Confidence: {confidence:.2f}")
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import cv2

# Load models
densenet_model = tf.keras.models.load_model(r'C:\Users\91874\Downloads\TB\best_model_unfrozen.h5')
yolo_model = YOLO(r'C:\Users\91874\Downloads\TB\best (6).pt')  # or .yaml if custom trained

# Classes
densenet_classes = ['healthy', 'sick', 'tb']
yolo_classes = ['ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']

def densenet_predict(image_path):
    """DenseNet classification"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((512, 512))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    predictions = densenet_model.predict(image_array)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    return densenet_classes[predicted_idx], confidence

def yolo_detect(image_path):
    """YOLO detection for TB regions"""
    results = yolo_model(image_path)
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

def draw_detections(image_path, detections):
    """Draw bounding boxes on image"""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Draw rectangle
        draw.rectangle(bbox, outline='red', width=3)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        draw.text((bbox[0], bbox[1]-20), label, fill='red')
    
    return image

def combined_inference(image_path):
    """Combined pipeline: DenseNet -> YOLO if TB detected"""
    print("Step 1: Running DenseNet classification...")
    classification, conf = densenet_predict(image_path)
    
    print(f"DenseNet Result: {classification} (confidence: {conf:.2f})")
    
    result = {
        'classification': classification,
        'classification_confidence': conf,
        'detections': None,
        'annotated_image': None
    }
    
    # If classified as TB, run YOLO detection
    if classification.lower() == 'tb':
        print("Step 2: TB detected! Running YOLO detection...")
        detections = yolo_detect(image_path)
        
        if detections:
            print(f"Found {len(detections)} TB regions:")
            for i, detection in enumerate(detections):
                print(f"  Region {i+1}: {detection['class']} (confidence: {detection['confidence']:.2f})")
            
            # Create annotated image
            annotated_image = draw_detections(image_path, detections)
            result['detections'] = detections
            result['annotated_image'] = annotated_image
        else:
            print("No TB regions detected by YOLO")
    else:
        print("No TB detected, skipping YOLO detection")
    
    return result

# Usage
if __name__ == "__main__":
    image_path = 'your_chest_xray.jpg'
    result = combined_inference(r'C:\Users\91874\Downloads\TB\tb0006.png')
    
    # Save annotated image if TB regions were detected
    if result['annotated_image']:
        result['annotated_image'].save('annotated_result.jpg')
        print("Annotated image saved as 'annotated_result.jpg'")
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['classification_confidence']:.2f}")
    
    if result['detections']:
        print(f"TB Regions Found: {len(result['detections'])}")
        for i, det in enumerate(result['detections']):
            print(f"  {i+1}. {det['class']} - Confidence: {det['confidence']:.2f}")