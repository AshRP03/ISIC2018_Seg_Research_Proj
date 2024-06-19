import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_ext='.jpg', mask_suffix='_segmentation.png'):
        self.image_dir = os.path.abspath(image_dir)
        self.mask_dir = os.path.abspath(mask_dir)
        self.transform = transform
        self.image_ext = image_ext
        self.mask_suffix = mask_suffix
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith(self.image_ext)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_base = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_base + self.mask_suffix
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Check if files exist
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

class ISICTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = ISICDataset('data/train/images', 'data/train/masks', transform, image_ext='.jpg', mask_suffix='_segmentation.png')
val_dataset = ISICDataset('data/val/images', 'data/val/masks', transform, image_ext='.jpg', mask_suffix='_segmentation.png')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)