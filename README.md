# Step 1: Install necessary libraries (run in terminal or command prompt)
!pip install torch torchvision
!pip install ultralytics
# Object Detection using YOLOv5 and OpenCV

# Import necessary libraries
import torch
import cv2
import matplotlib.pyplot as plt

# Step 2: Load the YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is a small model for fast inference

# Step 3: Define image path
image_path = "path_to_your_image.jpg"  # Replace with the actual image path

# Step 4: Perform object detection on the image
results = model("/content/360_F_65706597_uNm2SwlPIuNUDuMwo6stBd81e25Y8K8s.jpg")

# Step 5: Display the results (image with bounding boxes)
results.show()

# Step 6: Save the results (image with bounding boxes)
results.save()

# Step 7: Output the detected objects and their details
# Extracting bounding box coordinates, confidence scores, and class names
detections = results.pandas().xywh[0]  # Get detections as a Pandas DataFrame

# Loop through each detection and print its details
for index, detection in detections.iterrows():
    class_name = detection['name']
    confidence = detection['confidence']

    # Access bounding box coordinates using 'xcenter', 'ycenter', 'width', and 'height'
    x_center, y_center, width, height = detection['xcenter'], detection['ycenter'], detection['width'], detection['height']

    # Calculate xmin, ymin, xmax, ymax from xcenter, ycenter, width, and height
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)

    print(f"Detected Object: {class_name}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Bounding Box Coordinates: ({x1}, {y1}), ({x2}, {y2})\n")
