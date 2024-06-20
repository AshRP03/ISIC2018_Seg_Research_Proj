import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import ISICTestDataset
from unet_model import UNet
import matplotlib.pyplot as plt

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = UNet().to(device)
model.load_state_dict(torch.load('models/unet.pth'))
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create test dataset and dataloader
test_dataset = ISICTestDataset('data/test/images', transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Function to visualize predictions
def visualize(image, pred, img_name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[0].set_title(f'Image: {img_name}')
    ax[1].imshow(pred.squeeze().cpu().numpy(), cmap='gray')
    ax[1].set_title('Prediction')
    plt.show()

# Run the model on the test dataset
with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)  # Move inputs to the device
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5
        for i in range(images.size(0)):
            visualize(images[i], preds[i], img_names[i])