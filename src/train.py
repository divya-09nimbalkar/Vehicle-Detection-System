import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.data_pipeline import get_dataloader

def train_model(net, epochs=5):
    loader = get_dataloader()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        correct, total, running_loss = 0, 0, 0.0
        for imgs, labels in tqdm(loader):
            outputs = net(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader):.4f} | Acc: {correct/total:.2f}")

    torch.save(net.state_dict(), "models/vehicle_resnet18.pth")
    print("💾 Model saved to models/vehicle_resnet18.pth")
