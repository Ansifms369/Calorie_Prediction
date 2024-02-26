import streamlit as st
import subprocess
from PIL import Image
import os
import numpy as np
import re
from ultralytics import YOLO

classNames = ['Apple', 'Chapathi', 'Chicken Gravy', 'Fries', 'Idli', 'Pizza', 'Rice', 'Soda', 'Tomato', 'Vada', 'banana', 'burger']

food_densities = {
    'Apple': 0.5,
    'Chapathi': 0.6,
    'Chicken Gravy': 0.7,
    'Fries': 0.4,
    'Idli': 0.8,
    'Pizza': 0.5,
    'Rice': 0.7,
    'Soda': 1.0,
    'Tomato': 0.6,
    'Vada': 0.6,
    'banana': 0.5,
    'burger': 0.7
}

calories_per_100g = {
    'Apple': 52,
    'Chapathi': 297,
    'Chicken Gravy': 121,
    'Fries': 312,
    'Idli': 26,
    'Pizza': 266,
    'Rice': 130,
    'Soda': 42,
    'Tomato': 18,
    'Vada': 230,
    'banana': 89,
    'burger': 295
}

model = YOLO("runs/segment/train2/weights/best.pt")

def strip_escape_sequences(text):
    # Regular expression to match escape sequences
    escape_sequence = re.compile(r'\x1b\[\d+m')
    # Replace escape sequences with empty string
    return escape_sequence.sub('', text)

# Streamlit app title
st.title("Food Calorie Estimation")

# File upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to perform segmentation and calorie estimation
    if st.button('Perform Segmentation and Calorie Estimation'):
        results = model.predict(image)
        masks = results[0].masks.data

        masks = np.array(image.resize((masks.shape[1], masks.shape[0])))

        # Save uploaded image to disk
        uploaded_image_path = "uploaded_image.jpg"
        image.save(uploaded_image_path)

        # Define YOLO command
        yolo_command = [
            "yolo",
            "task=segment",
            "mode=predict",
            "model=runs/segment/train2/weights/best.pt",
            f"source={uploaded_image_path}",
            "save_txt=True"
        ]

        try:
            # Run YOLO command and capture output
            process = subprocess.Popen(yolo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                output_str = stdout.decode()
                # Extract the path from the output, removing escape sequences
                match = re.search(r'Results saved to (.+)', strip_escape_sequences(output_str))
                match1 = re.search(r'1 label saved to (.+)', strip_escape_sequences(output_str))
                if match:
                    # Extract the directory where segmented images are stored
                    segmented_images_dir = match.group(1).strip()
                    # Find any image file in the output directory
                    segmented_image_files = [f for f in os.listdir(segmented_images_dir) if os.path.isfile(os.path.join(segmented_images_dir, f))]
                    if segmented_image_files:
                        segmented_image_file = os.path.join(segmented_images_dir, segmented_image_files[0])
                        st.write("Segmented Image:")
                        segmented_image = Image.open(segmented_image_file)
                        st.image(segmented_image, caption='Segmented Image', use_column_width=True)
                    else:
                        st.write("No segmented image found in the output directory")
                else:
                    st.write("Segmented image path not found in output")

                if match1:
                    # Extract the directory where segmented images are stored
                    text_dir = match1.group(1).strip()
                    # Find any image file in the output directory
                    text_files = [f for f in os.listdir(text_dir) if os.path.isfile(os.path.join(text_dir, f))]
                    if text_files:
                        text_file = os.path.join(text_dir, text_files[0])
                        outs = open(text_file, "r").readlines()
                        for dt in outs:
                            index = int(dt.split()[0])  # Convert the first part of the string to an integer
                            class_name = classNames[index]
                    else:
                        st.write("No text")
                else:
                    st.write("No text path")
            else:
                st.error("Error running YOLO command:")
                st.error(stderr.decode())        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        selected_food_index = classNames.index(class_name)
        non_zero_pixels = np.count_nonzero(masks)
        surface_area = non_zero_pixels
        density = food_densities.get(class_name, 0.6)
        volume = surface_area * density
        calories = (volume / 100) * calories_per_100g.get(class_name, 0)
        st.write(f"This is {class_name} with {calories:.2f} Calories")
