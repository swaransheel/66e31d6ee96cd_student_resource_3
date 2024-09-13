import pandas as pd
from torch.utils.data import random_split
from src.utils import download_images
from src.constants import ALLOWED_UNITS
from src.sanity import check_output_format

def main():
    # Load and preprocess data
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    # Download images (if not already done)
    download_images(train_df)
    download_images(test_df)
    
    # Create datasets
    train_dataset = CustomDataset('dataset/train.csv', transform=transform)
    test_dataset = CustomDataset('dataset/test.csv', transform=transform, is_test=True)
    
    # Split train dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Initialize model
    model = EntityExtractionModel(num_classes=100)  # Adjust num_classes as needed
    
    # Train model
    trained_model = train_model(model, train_dataset, val_dataset)
    
    # Make predictions on test set
    predictions = predict(trained_model, test_dataset)
    
    # Post-process predictions
    formatted_predictions = post_process(predictions, ALLOWED_UNITS)
    
    # Generate output file
    generate_output(formatted_predictions, 'output.csv')

if __name__ == '__main__':
    main()
