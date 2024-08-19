from torch.utils.data import DataLoader
# Assuming this is renamed from CustomYOLODataset
from dataset import OxfordPetsDataset
import torchvision.transforms as transforms


def get_dataloaders(img_dir, ann_dir, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    dataset = OxfordPetsDataset(img_dir, ann_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    return dataloader
