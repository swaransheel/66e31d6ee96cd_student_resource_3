import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_dataset, val_dataset, num_epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        for images, entity_names, entity_values in train_loader:
            images = images.to(device)
            labels = torch.tensor([float(val.split()[0]) for val in entity_values]).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, entity_names, entity_values in val_loader:
                images = images.to(device)
                labels = torch.tensor([float(val.split()[0]) for val in entity_values]).to(device)
                
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader)}")
    
    return model

# Usage
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
trained_model = train_model(model, train_dataset, val_dataset)
