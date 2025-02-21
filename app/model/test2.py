# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# from app.model.DataPreparation import generate_cluster, pad_cluster, preprocess
# from app.model.isDeadCluster import isClusterDead
# from app.model.isSplitBrain import isSplitBrain
#
# class SplitBrainModel(nn.Module):
#     def __init__(self):
#         super(SplitBrainModel, self).__init__()
#         self.fc1 = nn.Linear(9 * 9 + 9, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 1)
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.sigmoid(self.fc4(x))
#         return x
#
# def train_model():
#     model = SplitBrainModel()
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#
#     for epoch in range(2):
#         total_loss = 0.0
#         for _ in range(1000000):
#             nodes, matrix = generate_cluster()
#             while isClusterDead(nodes, matrix):
#                 nodes, matrix = generate_cluster()
#
#             padded_nodes, padded_matrix = pad_cluster(nodes, matrix)
#             x_nodes = [2 if n == "A" else 3 if n == "B" else 4 for n in padded_nodes]
#             x_matrix = padded_matrix.flatten()
#             x_input = torch.tensor(x_nodes + x_matrix.tolist(), dtype=torch.float32)
#             x_input = (x_input - x_input.mean()) / (x_input.std() + 1e-6)
#             x_input = x_input.unsqueeze(0)
#
#             label = isSplitBrain(nodes, matrix)
#             y_target = torch.tensor([[label]], dtype=torch.float32)
#
#             optimizer.zero_grad()
#             output = model(x_input)
#             loss = criterion(output, y_target)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         avg_loss = total_loss / 100000
#         print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.6f}")
#         scheduler.step(avg_loss)
#
#     return model
#
# def save_model(model, filename="split_brain_model_revo.pth"):
#     torch.save(model.state_dict(), filename)
#
# def predict_neural_model(nodes, matrix):
#     model_path = 'split_brain_model_revo.pth'
#     print("NM __________________")
#
#     def load_model():
#         model = SplitBrainModel()
#         if os.path.exists(model_path):
#             model.load_state_dict(torch.load(model_path, weights_only=True))
#             model.eval()
#         else:
#             print("Модель не знайдена. Починається тренування...")
#             model = train_model()
#             save_model(model, model_path)
#         return model
#
#     x_input = preprocess(nodes, matrix)
#     x_input = torch.tensor(x_input, dtype=torch.float32)
#     x_input = (x_input - x_input.mean()) / (x_input.std() + 1e-6)
#     x_input = x_input.unsqueeze(0)
#
#     model = load_model()
#
#     with torch.no_grad():
#         prediction = model(x_input).item()
#         return prediction
#
# def teach_neural_model(nodes, matrix):
#     model_path = "split_brain_model_good_2.pth"
#
#     def load_model():
#         model = SplitBrainModel()
#         if os.path.exists(model_path):
#             model.load_state_dict(torch.load(model_path))
#             print("Модель успішно завантажена.")
#         else:
#             print("Модель не знайдена. Створення нової моделі...")
#         return model
#
#     model = load_model()
#     model.train()
#
#     x_input = preprocess(nodes, matrix)
#     x_input = torch.tensor(x_input, dtype=torch.float32)
#     x_input = (x_input - x_input.mean()) / (x_input.std() + 1e-6)
#     x_input = x_input.unsqueeze(0)
#
#     label = isSplitBrain(nodes, matrix)
#     y_target = torch.tensor([[label]], dtype=torch.float32)
#
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
#
#     optimizer.zero_grad()
#     output = model(x_input)
#     loss = criterion(output, y_target)
#     loss.backward()
#     optimizer.step()
#
#     torch.save(model.state_dict(), model_path)
#     print(f"Модель успішно оновлена та збережена. Втрата: {loss.item():.6f}")
#     return loss.item()
