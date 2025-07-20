import os
import pickle

from sklearn.metrics import log_loss
from tqdm import tqdm
from catboost import CatBoostClassifier

from app.model.DataPreparation import *


def train_model():
    model = CatBoostClassifier(
        iterations=1000,
        depth=10,
        learning_rate=0.1,
        loss_function='Logloss',
        verbose=False,
        thread_count=6
    )
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

    print("Навчання моделі...")
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
    print("CB __________________")
    if isSingleType(nodes):
        return 0
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    return model.predict_proba(x_input)[0, 1]


def teach_cb(nodes, matrix):
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

    with open("split_brain_model_cb.pkl", "wb") as f:
        pickle.dump(model, f)

    return current_loss * 1000
