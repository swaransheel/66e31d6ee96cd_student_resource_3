import torch
import torch.nn as nn
import torchvision.models as models

class EntityExtractionModel(nn.Module):
    def __init__(self, num_classes):
        super(EntityExtractionModel, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Usage
model = EntityExtractionModel(num_classes=100)  # Adjust num_classes based on your dataset
