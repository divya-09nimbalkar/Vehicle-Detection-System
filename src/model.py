import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=2):  # Vehicle / Non-Vehicle
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net
