import torch.nn as nn
from torchvision import models


def get_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

