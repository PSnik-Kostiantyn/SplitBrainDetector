import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier

from app.model.DataPreparation import generate_cluster, preprocess, isClusterDead, isSplitBrain


def train_model():
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    #learning_rate=0.1, max_depth=5
    X_train, y_train = [], []
    for _ in range(200000):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X_train.append(preprocess(nodes, matrix))
        y_train.append(isSplitBrain(nodes, matrix))
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
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    return model.predict_proba(x_input)[0, 1]


def teach_gb(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    label = isSplitBrain(nodes, matrix)
    model.fit(x_input, [label])
    with open("split_brain_model_gb.pkl", "wb") as f:
        pickle.dump(model, f)
    return label
