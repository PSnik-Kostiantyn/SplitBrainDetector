import os
import pickle
from catboost import CatBoostClassifier

from app.model.DataPreparation import *

def train_model():
    model = CatBoostClassifier(iterations=150, depth=8, learning_rate=0.05, loss_function='Logloss', verbose=0)
    X_train, y_train = [], []
    for _ in range(1000000):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X_train.append(preprocess(nodes, matrix))
        y_train.append(isSplitBrain(nodes, matrix))
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
    if (isSingleType(nodes)):
        return 0
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    return model.predict_proba(x_input)[0, 1]


def teach_cb(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    label = isSplitBrain(nodes, matrix)
    model.fit(x_input, [label])
    with open("split_brain_model_cb.pkl", "wb") as f:
        pickle.dump(model, f)
    return label
