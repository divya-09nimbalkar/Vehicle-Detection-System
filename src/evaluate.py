import torch
import torchvision.models as models

def load_model(path="models/vehicle_resnet18.pth", num_classes=2):
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    net.load_state_dict(torch.load(path))
    net.eval()
    print(f"✅ Model loaded from {path}")
    return net
