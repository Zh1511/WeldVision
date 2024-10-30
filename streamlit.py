import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

# Title of the Streamlit App
st.title("WeldVision")

# Display disclaimer message
st.markdown("**Disclaimer: This model is for academic purposes only and not intended for commercial use.**")

# Function to load the model and check if it's using GPU or CPU
@st.cache_resource
def load_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO("trained.pt")  # Load the model
        model.to(device)  # Move model to GPU if available or CPU
        st.write("Model loaded successfully and working.")
        return model
    except RuntimeError as e:
        st.write(f"RuntimeError: {e}")
        return None
    except Exception as e:
        st.write(f"An error occurred while loading the model: {e}")
        return None

# Load the model
model = load_model()

# Function to resize the image to 640x640
def resize_image(image, size=(640, 640)):
    return image.resize(size)

# Upload an image using Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to draw customized bounding boxes with different colors based on labels
from PIL import Image, ImageDraw, ImageFont

# Function to draw customized bounding boxes with transparent overlays
def draw_custom_boxes(image, boxes, labels, confidences):
    # Create a separate overlay image for transparency
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))  # Fully transparent

    # Create a drawing context on the overlay image
    draw = ImageDraw.Draw(overlay)

    # Load a basic font
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Use a specific font if available
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    for box, label, conf in zip(boxes, labels, confidences):
        # Define box coordinates
        x1, y1, x2, y2 = box

        # Determine color based on label with transparency (RGBA)
        if label == 'bad weld':
            box_color = (128, 0, 128, 50)  # Purple with transparency
        elif label == 'defect':
            box_color = (255, 0, 0, 50)    # Red with transparency
        else:
            box_color = (0, 255, 0, 50)    # Green with transparency

        # Draw the filled bounding box on the overlay
        draw.rectangle([x1, y1, x2, y2], fill=box_color)

        # Draw the outline on the overlay
        draw.rectangle([x1, y1, x2, y2], outline=(box_color[0], box_color[1], box_color[2]), width=3)

        # Draw text label and confidence on the main image
        text_label = f"{label} {conf:.2f}"
        image_draw = ImageDraw.Draw(image)
        image_draw.text((x1, y1), text_label, fill=(255, 255, 255), font=font)

    # Composite the overlay onto the original image to apply transparency
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    return image.convert("RGB")  # Convert back to RGB for displaying


# Function to display detection results
def display_results(boxes, labels, confidences):
    st.write("**Detection Results**")
    for i, (label, conf) in enumerate(zip(labels, confidences)):
        st.write(f"Object {i + 1}: **{label}**, Confidence: **{conf:.2f}**")

# Function to process the image
def process_image(image, confidence_threshold=0.5):
    if model is None:
        st.write("Model is not loaded.")
        return

    try:
        # Resize the image
        resized_image = resize_image(image)
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
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Assuming the model provides x1, y1, x2, y2 format
                    width, height = resized_image.size

                    # Clamp coordinates to stay within image dimensions
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    # Calculate box width and height for additional validation
                    box_width = x2 - x1
                    box_height = y2 - y1

                    # Skip adding the box if it exceeds 70% of the image dimensions
                    if box_width > 0.7 * width or box_height > 0.7 * height:
                        print("Skipping bounding box as it is too large")
                        continue

                    boxes.append([x1, y1, x2, y2])
                except Exception as coord_error:
                    st.write(f"Error processing bounding box coordinates: {coord_error}")
                    continue

                # Get the class name from model
                labels.append(model.names[int(box.cls)])  # Class name from model
                confidences.append(float(box.conf))  # Confidence score

        # Draw the custom bounding boxes
        if boxes:
            annotated_image = draw_custom_boxes(resized_image.copy(), boxes, labels, confidences)
            st.image(annotated_image, caption="Detected Objects with Custom Boxes", use_column_width=True)
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
