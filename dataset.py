import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms

class OxfordPetsDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        annotation_path = os.path.join(self.annotation_dir, img_file.replace('.jpg', '.xml'))
        
        image = Image.open(img_path).convert('RGB')
        target = []
        if os.path.exists(annotation_path):
            target = self.parse_voc_annotation(annotation_path, image.size)
        
        if len(target) == 0:
            target = np.zeros((0, 5), dtype=np.float32)
        else:
            target = np.array(target, dtype=np.float32)
        
        target = torch.tensor(target, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

    def parse_voc_annotation(self, annotation_file, img_size):
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        
        img_width, img_height = img_size
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            cls = obj.find('name').text
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            class_id = self.class_name_to_id(cls)
            
            yolo_annotations.append([class_id, x_center, y_center, width, height])
        
        return yolo_annotations

    def class_name_to_id(self, class_name):
        class_map = {'cat': 0, 'dog': 1}
        return class_map.get(class_name, -1)
