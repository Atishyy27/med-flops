import torch.nn as nn
from torchvision.models import resnet18

def get_model(name="resnet18", num_classes=2):
    if name == "resnet18":
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model type: {name}")
