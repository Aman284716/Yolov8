import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw
import torch.nn.functional as F


def yolo_loss(pred, target):
    # Example loss components
    bbox_loss = F.mse_loss(pred[:, :4], target[:, :4])
    obj_loss = F.binary_cross_entropy_with_logits(pred[:, 4], target[:, 4])
    class_loss = F.cross_entropy(pred[:, 5:], target[:, 5:])

    total_loss = bbox_loss + obj_loss + class_loss
    return total_loss


def visualize_predictions(image, outputs):
    # Assuming outputs contain bounding boxes, class labels, and confidence scores
    # Format: [batch_size, num_boxes, 4 (bbox), 1 (objectness), num_classes]

    # Example: Convert outputs to numpy for easier manipulation
    image_np = np.array(image)
    # Adjust according to your output structure
    boxes = outputs[0]['boxes'].detach().numpy()
    # Adjust according to your output structure
    scores = outputs[0]['scores'].detach().numpy()
    # Adjust according to your output structure
    labels = outputs[0]['labels'].detach().numpy()

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        # Convert box coordinates from [0, 1] to image pixel values
        box = [int(coord) for coord in box]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f'{label}: {score:.2f}', fill="red")

    plt.imshow(image)
    plt.axis('off')
    plt.show()
