import os
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from tqdm import tqdm

from app.model.DataPreparation import generate_cluster, preprocess, isClusterDead, isSplitBrain, isSingleType


def train_model():
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=2,
        random_state=42
    )

    #learning_rate=0.1, max_depth=5
    X_train, y_train = [], []

    for _ in tqdm(range(900000), desc="Generating normal data"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X_train.append(preprocess(nodes, matrix))
        y_train.append(isSplitBrain(nodes, matrix))

    for _ in tqdm(range(500000), desc="Generating additional split-brain cases"):
        nodes, matrix = generate_cluster()
        while not isSplitBrain(nodes, matrix):
            nodes, matrix = generate_cluster()
        sample = preprocess(nodes, matrix)
        for _ in range(2):
            X_train.append(sample)
            y_train.append(1)

    model.fit(X_train, y_train)
    with open("split_brain_model_gb.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

def load_model():
    model_path = "split_brain_model_gb.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        return train_model()

def predict_gb(nodes, matrix):
    print("GB __________________")
    if isSingleType(nodes):
        return 0
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    return model.predict_proba(x_input)[0, 1]

def teach_gb(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    label = isSplitBrain(nodes, matrix)

    other_label = 1 - label
    dummy_input = np.zeros_like(x_input)

    X = np.vstack([x_input, dummy_input])
    y = [label, other_label]

    model.fit(X, y)

    proba = model.predict_proba(x_input)
    current_loss = log_loss([label], proba, labels=[0, 1])

    with open("split_brain_model_gb.pkl", "wb") as f:
        pickle.dump(model, f)

    return current_loss * 1000000