import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, image_dir, img_size):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), int(row["label"])

def get_data_loaders(manifest_path: str, img_size: int, batch_size: int):
    df = pd.read_csv(manifest_path)

    # Stratified split: train → val/test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42
    )

    image_dir = os.path.dirname(manifest_path)

    train_dataset = PneumoniaDataset(train_df, image_dir, img_size)
    val_dataset   = PneumoniaDataset(val_df, image_dir, img_size)
    test_dataset  = PneumoniaDataset(test_df, image_dir, img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
