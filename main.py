# # import matplotlib.pyplot as plt
# # import numpy as np
# # import pandas as pd
# # import seaborn as sns
# # import os
# # from tensorflow.keras.preprocessing.image import load_img, img_to_array
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
# # from tensorflow.keras.models import Model,Sequential
# # from tensorflow.keras.optimizers import Adam,SGD,RMSprop
#
# # Import necessary libraries
# import torch
# from PIL import Image
# import requests
# import matplotlib.pyplot as plt
# from facenet_pytorch import MTCNN
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#
# # Load the pre-trained YOLOv5 model (e.g., face detection or general object detection)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is a small, efficient model
#
# # Load an image (either from a URL or a local file)
# image_url = "F:/bala.jfif"  # Replace with your image URL or path
# image = Image.open(image_url)
#
# # Run inference
# results = model(image)
#
# # Print results
# results.print()  # Prints results to the console (detected classes and their confidence scores)
# results.show()   # Display the image with bounding boxes for detected objects (including faces and persons)
#
#
# # Load the image
# image_path = "F:/bala.jfif"
# image = Image.open(image_path)
#
# # Initialize MTCNN for face detection
# mtcnn = MTCNN(keep_all=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#
# # Detect faces
# boxes, _ = mtcnn.detect(image)
#
# # Plot the image with bounding boxes
# plt.figure(figsize=(8, 8))
# plt.imshow(image)
# ax = plt.gca()
#
# # Draw bounding boxes
# if boxes is not None:
#     for box in boxes:
#         rect = plt.Rectangle(
#             (box[0], box[1]),
#             box[2] - box[0],
#             box[3] - box[1],
#             linewidth=2,
#             edgecolor='red',
#             facecolor='none',
#         )
#         ax.add_patch(rect)
#
# plt.axis("off")
# plt.show()

import cv2
import os
from datetime import datetime
import torch

# Load a pre-trained YOLOv5 model from the Ultralytics YOLOv5 repository
# Make sure to have YOLOv5 installed
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 small model
model.classes = [0]  # YOLO class for people (players)

# Define input video path and output folder for images
video_path = 'F:/LUMS/EmotionDetection/bb2.mp4'
output_folder = 'extracted_players'
os.makedirs(output_folder, exist_ok=True)

# Set up video capture
cap = cv2.VideoCapture(video_path)
frame_rate = 1  # Capture every frame
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the detection every N frames (for efficiency)
    if frame_id % (frame_rate * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
        results = model(frame)
        for i, det in enumerate(results.xyxy[0]):  # Iterate over detections
            if det[5] == 0:  # Class 0 is 'person' in YOLO
                x1, y1, x2, y2 = map(int, det[:4])  # Coordinates of the bounding box
                player_img = frame[y1:y2, x1:x2]  # Crop the player image

                # Save the cropped player image with timestamp
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                player_img_path = os.path.join(output_folder, f'player_{frame_id}_{timestamp}.jpg')
                cv2.imwrite(player_img_path, player_img)

    frame_id += 1

# Release resources
cap.release()
print("Extraction completed.")
