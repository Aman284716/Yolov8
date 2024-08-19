from utils import visualize_predictions
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import YOLOv8
from utils import transform

# Load the trained model
model = YOLOv8(nc=80)
model.load_state_dict(torch.load('yolov8.pth'))
model.eval()

# Load and prepare the image
image_path = 'path_to_image.jpg'
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    outputs = model(image_tensor)

# Process and visualize outputs (implement this in utils.py)
visualize_predictions(image, outputs)
