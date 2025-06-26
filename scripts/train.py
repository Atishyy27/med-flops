from pneumonia_detector.data_loader import PneumoniaDataset
from pneumonia_detector.models import get_model
from pneumonia_detector.trainer import train
from torch.utils.data import DataLoader

dataset = PneumoniaDataset(csv_path="data/manifest.csv", image_root="data/images")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = get_model("resnet18")
train(model, dataloader, epochs=3, device="cuda")
