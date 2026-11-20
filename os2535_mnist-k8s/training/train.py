
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTCNN

torch.manual_seed(42)

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DATA_DIR = os.getenv("DATA_DIR", "/tmp/mnist")
EPOCHS = int(os.getenv("EPOCHS", "3"))

os.makedirs(MODEL_DIR, exist_ok=True)

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    DATA_DIR, train=True, download=True,
    transform=transform
)
test_data = datasets.MNIST(
    DATA_DIR, train=False, download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True
)
test_loader = DataLoader(
    test_data, batch_size=256
)

device = torch.device("cpu")
model = MNISTCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, "
        f"Loss: {total_loss / len(train_loader):.4f}",
        flush=True
    )

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test accuracy: {accuracy:.2f}%", flush=True)

# Save atomically to avoid exposing a partial checkpoint.
temporary_path = os.path.join(MODEL_DIR, "model.tmp")
model_path = os.path.join(MODEL_DIR, "model.pt")

torch.save(model.state_dict(), temporary_path)
os.replace(temporary_path, model_path)

print(f"Model saved at {model_path}", flush=True)
