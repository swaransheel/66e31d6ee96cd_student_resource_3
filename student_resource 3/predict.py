import torch
from torch.utils.data import DataLoader
import pytesseract
from src.constants import ALLOWED_UNITS

def predict(model, test_dataset, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    predictions = []
    
    with torch.no_grad():
        for images, indices, entity_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions.extend(list(zip(indices, outputs.cpu().numpy(), entity_names)))
    
    return predictions

def post_process(predictions, allowed_units):
    processed_predictions = []
    
    for index, output, entity_name in predictions:
        # Use OCR to extract text from the image
        image = test_dataset[index][0].permute(1, 2, 0).numpy()
        extracted_text = pytesseract.image_to_string(image)
        
        # Find relevant value in extracted text
        # This is a simplified example; you'd need more robust parsing
        value = None
        for word in extracted_text.split():
            try:
                value = float(word)
                break
            except ValueError:
                continue
        
        if value is not None:
            unit = allowed_units[entity_name][0]  # Use first allowed unit as default
            processed_predictions.append((index, f"{value:.2f} {unit}"))
        else:
            processed_predictions.append((index, ""))
    
    return processed_predictions

# Usage
predictions = predict(trained_model, test_dataset)
formatted_predictions = post_process(predictions, ALLOWED_UNITS)
