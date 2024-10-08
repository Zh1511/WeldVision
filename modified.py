import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

# Title of the Streamlit App
st.title("Welding Detection with YOLOv8")

# Display disclaimer message
st.markdown("**Disclaimer: This model is for academic purposes only and not intended for commercial use.**")


# Function to check if the model is loaded successfully
def check_model(model):
    try:
        # Test the model with a dummy input to ensure it's loaded correctly
        dummy_image = Image.new('RGB', (640, 640), color='white')  # Create a dummy image
        results = model(dummy_image, conf=0.25)  # Run a test inference
        if results:
            st.write("Model loaded successfully and working.")
        else:
            st.write("Model loaded but failed to perform inference.")
    except Exception as e:
        st.write(f"Model loading or inference failed: {e}")


# Function to load the selected model and check if it's using GPU or CPU
@st.cache_resource
def load_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO("yolov8m.pt")  # Load the yolov8m model
        model.to(device)  # Move model to GPU if available or CPU
        check_model(model)  # Check if the model is loaded and working
        return model
    except RuntimeError as e:
        st.write(f"RuntimeError: {e}")
    except Exception as e:
        st.write(f"An error occurred: {e}")


# Load the model (using yolov8m by default)
model = load_model()

# Define a dictionary to map YOLO class names to custom labels
class_mapping = {
    'class_1': 'bad weld',   # Replace 'class_1' with the actual YOLO class name
    'class_2': 'defect',     # Replace 'class_2' with the actual YOLO class name
    'class_3': 'good weld',  # Replace 'class_3' with the actual YOLO class name
    # Add more mappings as necessary depending on the YOLO model's class names
}

# Function to resize the image to 640x640
def resize_image(image, size=(640, 640)):
    return image.resize(size)


# Upload an image using Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


# Function to draw customized bounding boxes with different colors based on labels
def draw_custom_boxes(image, boxes, labels, confidences):
    # Create a drawing context
    draw = ImageDraw.Draw(image, "RGBA")

    # Load a basic font
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Use a specific font if available
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    for box, label, conf in zip(boxes, labels, confidences):
        # Define box coordinates
        x1, y1, x2, y2 = box

        # Determine color based on label
        if label == 'bad weld':
            box_color = (128, 0, 128)  # Purple outline
            fill_color = (128, 0, 128, 50)  # Transparent purple fill
        elif label == 'defect':
            box_color = (255, 0, 0)  # Red outline
            fill_color = (255, 0, 0, 50)  # Transparent red fill
        else:
            box_color = (0, 255, 0)  # Green outline for other categories
            fill_color = (0, 255, 0, 50)  # Transparent green fill

        # Draw the bounding box with the selected color
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        # Draw text label and confidence
        text_label = f"{label} {conf:.2f}"
        draw.text((x1, y1), text_label, fill=(255, 255, 255, 255), font=font)

    return image


# Function to display detection results
def display_results(boxes, labels, confidences):
    st.write("**Detection Results**")
    for i, (label, conf) in enumerate(zip(labels, confidences)):
        st.write(f"Object {i + 1}: **{label}**, Confidence: **{conf:.2f}**")


# Function to process the image
def process_image(image, confidence_threshold=0.25):
    try:
        # Resize the image
        resized_image = resize_image(image)

        # Display the resized image in Streamlit
        st.image(resized_image, caption='Resized Image (640x640)', use_column_width=True)

        # Run model prediction
        results = model(resized_image, conf=confidence_threshold)  # Directly use PIL image

        # Extract bounding boxes, labels, and confidences
        boxes = []
        labels = []
        confidences = []

        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                boxes.append([x1, y1, x2, y2])
                labels.append(model.names[int(box.cls)])  # Class name
                confidences.append(float(box.conf))  # Confidence score

        # Draw the custom bounding boxes
        if boxes:
            annotated_image = draw_custom_boxes(resized_image.copy(), boxes, labels, confidences)
            st.image(annotated_image, caption="Detected Objects with Custom Boxes", use_column_width=True)
            
            # Display results below the annotated image
            display_results(boxes, labels, confidences)
        else:
            st.write("No objects detected.")

    except Exception as e:
        st.write(f"An error occurred: {e}")


# Only process the image if it has been uploaded
if uploaded_file is not None:
    try:
        # Load the uploaded image
        image = Image.open(uploaded_file)

        # Run the object detection
        process_image(image)

    except Exception as e:
        st.write(f"An error occurred while processing the image: {e}")
