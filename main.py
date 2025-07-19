import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AquaVision: Sea Animal Detection",
    page_icon="🐠",
    layout="wide"
)

st.title("🐠 AquaVision: Real-Time Sea Animal Detection")
st.markdown("Upload an image or video to detect sea animals using AI-powered computer vision.")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model():
    """Load YOLO model with caching for better performance"""
    try:
        # Check if model file exists
        model_path = 'best.pt'
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the repository root.")
            return None
        
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        logger.error(f"Model loading error: {e}")
        return None

def predict_and_save_image(model, input_path, output_path):
    """Process image with YOLO detection"""
    try:
        results = model.predict(input_path, device='cpu', verbose=False)
        image = cv2.imread(input_path)
        
        if image is None:
            raise ValueError("Could not read image file")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_count = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls] if hasattr(model, 'names') else f'class {cls}'

                    # Dynamic thresholds based on class
                    threshold = 0.75 if class_name.lower() == 'fish' else 0.4

                    if confidence > threshold:
                        # Draw bounding box
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add label with confidence
                        label = f'{class_name} {confidence*100:.1f}%'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Background rectangle for text
                        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        
                        cv2.putText(image, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        detection_count += 1
        
        # Convert back to BGR for saving
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        
        return output_path, detection_count
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        st.error(f"Error processing image: {e}")
        return None, 0

def predict_and_plot_video(model, video_path, output_path, progress_bar):
    """Process video with YOLO detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        total_detections = 0
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu', verbose=False)
            
            frame_detections = 0
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls] if hasattr(model, 'names') else f'class {cls}'

                        # Dynamic thresholds
                        threshold = 0.8 if class_name.lower() == 'fish' else 0.5

                        if confidence > threshold:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            label = f'{class_name} {confidence*100:.1f}%'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (0, 255, 0), -1)
                            
                            cv2.putText(frame, label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            frame_detections += 1

            out.write(frame)
            total_detections += frame_detections
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress, text=f"Processing frame {frame_count}/{total_frames}")

        cap.release()
        out.release()
        
        return output_path, total_detections

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        st.error(f"Error processing video: {e}")
        return None, 0

def main():
    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_model()
    
    model = st.session_state.model
    
    if model is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
        help="Supported formats: Images (JPG, PNG, BMP) and Videos (MP4, AVI, MOV, MKV)"
    )

    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            file_type = "Image" if uploaded_file.type.startswith('image') else "Video"
            st.metric("Type", file_type)

        # Process file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            input_path = os.path.join(temp_dir, uploaded_file.name)
            output_filename = f"detected_{uploaded_file.name}"
            output_path = os.path.join(temp_dir, output_filename)

            try:
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Determine file type and process
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                    st.subheader("🎬 Video Processing")
                    progress_bar = st.progress(0, text="Starting video processing...")
                    
                    with st.spinner("Processing video... This may take a while."):
                        result_path, detection_count = predict_and_plot_video(
                            model, input_path, output_path, progress_bar
                        )
                    
                    if result_path:
                        st.success(f"✅ Processing complete! Found {detection_count} detections.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Original Video")
                            st.video(uploaded_file)
                        with col2:
                            st.subheader("Processed Video")
                            with open(result_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Processed Video",
                                    data=f.read(),
                                    file_name=output_filename,
                                    mime="video/mp4"
                                )
                
                elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
                    st.subheader("🖼️ Image Processing")
                    
                    with st.spinner("Analyzing image..."):
                        result_path, detection_count = predict_and_save_image(
                            model, input_path, output_path
                        )
                    
                    if result_path:
                        st.success(f"✅ Analysis complete! Found {detection_count} detections.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Original Image")
                            st.image(uploaded_file, use_column_width=True)
                        with col2:
                            st.subheader("Detected Objects")
                            st.image(result_path, use_column_width=True)
                        
                        # Download button for processed image
                        with open(result_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Processed Image",
                                data=f.read(),
                                file_name=f"detected_{uploaded_file.name}",
                                mime=f"image/{file_extension[1:]}"
                            )

            except Exception as e:
                st.error(f"Error processing file: {e}")
                logger.error(f"File processing error: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        "🐠 **AquaVision** - Powered by YOLO deep learning model for marine life detection"
    )

if __name__ == "__main__":
    main()
