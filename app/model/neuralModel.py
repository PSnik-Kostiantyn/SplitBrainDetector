import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from app.model.isDeadCluster import isClusterDead
from app.model.isSplitBrain import isSplitBrain

def generate_data(num_nodes):
    while True:
        nodes = [random.choice(['A', 'B', 'C']) + str(i) for i in range(num_nodes)]
        matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes))
        np.fill_diagonal(matrix, 0)

        if isClusterDead(nodes, matrix):
            continue

        split_brain = isSplitBrain(nodes, matrix)
        return nodes, matrix, int(split_brain)

def generate_dataset(samples=70000, max_nodes=5):
    X, y = [], []
    max_features = max_nodes + max_nodes * max_nodes

    for _ in range(samples):
        num_nodes = 5
        nodes, matrix, split_brain = generate_data(num_nodes)

        node_types = [ord(node[0]) - ord('A') + 1 for node in nodes]
        flattened_matrix = matrix.flatten().tolist()

        features = node_types + flattened_matrix

        padded_features = features + [-1] * (max_features - len(features))
        X.append(padded_features)
        y.append(split_brain)

    return np.array(X, dtype=float), np.array(y)

class SplitBrainDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SplitBrainModel(nn.Module):
    def __init__(self, input_size):
        super(SplitBrainModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def predict_neural_model(nodes, matrix):
    model_path = 'split_brain_model.pth'
    num_nodes = len(nodes)
    node_types = [ord(node[0]) - ord('A') for node in nodes]

    if isinstance(matrix, list):
        matrix = np.array(matrix)

    flattened_matrix = matrix.flatten().tolist()
    features = node_types + flattened_matrix
    input_size = len(features)

    if not os.path.exists(model_path):
        print("Start learning")
        X, y = generate_dataset(samples=70000, max_nodes=5)

        dataset = SplitBrainDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = SplitBrainModel(input_size=input_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Навчання
        model.train()
        for epoch in range(10):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

        # Збереження моделі
        torch.save(model.state_dict(), model_path)
        print("Model saved")
    else:
        model = SplitBrainModel(input_size=input_size)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    with torch.no_grad():
        input_tensor = torch.tensor([features], dtype=torch.float32)
        prediction = model(input_tensor).item()
        return prediction
