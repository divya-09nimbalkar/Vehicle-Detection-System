from src.model import build_model
from src.train import train_model
from src.evaluate import load_model

def run():
    print("🚗 Vehicle Detection System")
    net = build_model()
    train_model(net, epochs=2)
    net = load_model()

if __name__ == "__main__":
    run()
