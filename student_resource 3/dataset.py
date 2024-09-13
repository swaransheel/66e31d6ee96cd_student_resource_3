import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from src.utils import download_images

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_test = is_test
        
        # Download images
        download_images(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        if not self.is_test:
            entity_name = self.data.iloc[idx]['entity_name']
            entity_value = self.data.iloc[idx]['entity_value']
            return image, entity_name, entity_value
        else:
            return image, self.data.iloc[idx]['index'], self.data.iloc[idx]['entity_name']

# Usage
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CustomDataset('dataset/train.csv', transform=transform)
test_dataset = CustomDataset('dataset/test.csv', transform=transform, is_test=True)
