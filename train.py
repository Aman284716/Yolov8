import torch
import torch.optim as optim
from model import YOLOv8
from dataloader import get_dataloaders
from utils import yolo_loss

def main():
    # Load the dataset
    img_dir = 'data/images'
    ann_dir = 'data/annotations'
    dataloader = get_dataloaders(img_dir, ann_dir)

    # Initialize the model
    model = YOLOv8(nc=2)  # Adjust `nc` for the number of classes
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in dataloader:
            # Move data to GPU if available
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()
                model = model.cuda()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = yolo_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')

    # Save the trained model
    torch.save(model.state_dict(), 'yolov8.pth')

if __name__ == '__main__':
    main()
