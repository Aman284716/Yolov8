import torch
from PIL import Image
from model import YOLOv8
from dataloader import get_dataloaders
from utils import visualize_predictions

# Load the trained model
model = YOLOv8(nc=2)
model.load_state_dict(torch.load('yolov8.pth'))
model.eval()

# Load the dataset
img_dir = 'data/images'
ann_dir = 'data/annotations'
# Batch size 1 for testing
dataloader = get_dataloaders(img_dir, ann_dir, batch_size=1)

# Test loop
for images, _ in dataloader:
    with torch.no_grad():
        outputs = model(images)
        visualize_predictions(images[0], outputs)
        break
