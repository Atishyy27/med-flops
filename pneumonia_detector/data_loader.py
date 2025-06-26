import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class PneumoniaDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.df.iloc[idx]['image'])
        label = self.df.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), label
