import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pneumonia_detector.data_loader import PneumoniaDataset
from pneumonia_detector.models import get_model
from pneumonia_detector.trainer import train
from torch.utils.data import DataLoader
import torch

dataset = PneumoniaDataset("data/manifest.csv", "data/images")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = get_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
train(model, dataloader, device)
