import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    densenet121, DenseNet121_Weights
)

def build_model(model_name: str, pretrained: bool, num_classes: int = 2):
    """
    Build and return a model given its name and pretrained flag.
    """
    name = model_name.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name in ("densenet121", "densenet-121"):
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"[models] Unknown model name '{model_name}'")

    return model
