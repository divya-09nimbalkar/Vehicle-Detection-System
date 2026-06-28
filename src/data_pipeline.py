from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder("data/processed", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
