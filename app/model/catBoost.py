import random
import os
import numpy as np
import pickle
from catboost import CatBoostClassifier

from app.model.gradientBoosting import generate_cluster, isClusterDead_2, isSplitBrain_2
from app.model.neuralModel import pad_cluster

def preprocess(nodes, matrix):
    max_nodes = 9
    padded_nodes, padded_matrix = pad_cluster(nodes, matrix, max_nodes)
    x_nodes = [1 if n == "A" else 2 if n == "B" else 3 for n in padded_nodes]
    x_matrix = padded_matrix.flatten()
    return np.array(x_nodes + x_matrix.tolist()) / 3.0

def train_model():
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=0)
    X_train, y_train = [], []

    for _ in range(100000):
        nodes, matrix = generate_cluster()
        while isClusterDead_2(nodes, matrix):
            nodes, matrix = generate_cluster()
        X_train.append(preprocess(nodes, matrix))
        y_train.append(isSplitBrain_2(nodes, matrix))

    model.fit(X_train, y_train)
    with open("split_brain_model_cb.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

def load_model():
    model_path = "split_brain_model_cb.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        return train_model()

def predict_cb(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    return model.predict_proba(x_input)[0, 1]

def teach_cb(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    label = isSplitBrain_2(nodes, matrix)
    model.fit(x_input, [label])
    with open("split_brain_model_cb.pkl", "wb") as f:
        pickle.dump(model, f)
    return label
