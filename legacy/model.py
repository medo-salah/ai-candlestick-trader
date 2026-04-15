import torch
import torch.nn as nn
import torchvision.models as models

def create_model(model_name: str, num_classes: int, pretrained: bool = False):
    model_name = model_name.lower()
    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        else:
            model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    # fallback
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
