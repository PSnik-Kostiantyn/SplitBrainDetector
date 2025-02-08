import random
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

from app.model.isDeadCluster import isClusterDead
from app.model.isSplitBrain import isSplitBrain
from app.model.neuralModel import generate_cluster


def pad_cluster(nodes, matrix, max_nodes=9):
    padded_nodes = nodes + [-1] * (max_nodes - len(nodes))
    padded_matrix = np.full((max_nodes, max_nodes), -1)
    padded_matrix[:len(matrix), :len(matrix)] = matrix
    return padded_nodes, padded_matrix

def preprocess(nodes, matrix):
    max_nodes = 9
    padded_nodes, padded_matrix = pad_cluster(nodes, matrix, max_nodes)
    x_nodes = [1 if n == "A" else 2 if n == "B" else 3 for n in padded_nodes]
    x_matrix = padded_matrix.flatten()
    return np.array(x_nodes + x_matrix.tolist()) / 3.0

def train_model():
    print("Start learning")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    x_train, y_train = [], []

    for _ in range(100000):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        x_train.append(preprocess(nodes, matrix))
        y_train.append(isSplitBrain(nodes, matrix))

    model.fit(x_train, y_train)
    print("Finished learning")
    with open("split_brain_model_rf.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

def load_model():
    model_path = "split_brain_model_rf.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        return train_model()

def predict_rf(nodes, matrix):
    model = load_model()
    print("Predicting")
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    print("Ready")
    return model.predict_proba(x_input)[0, 1]

def teach_rf(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    label = isSplitBrain(nodes, matrix)
    model.fit(x_input, [label])
    with open("split_brain_model_rf.pkl", "wb") as f:
        pickle.dump(model, f)
    return label
