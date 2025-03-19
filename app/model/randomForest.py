import os
import pickle
import time
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from app.model.DataPreparation import generate_cluster, preprocess, isClusterDead, isSplitBrain, isSingleType


def train_model():
    print("Start learning")
    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    x_train, y_train = [], []

    for _ in tqdm(range(500000), desc="Generating normal data"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        x_train.append(preprocess(nodes, matrix))
        y_train.append(isSplitBrain(nodes, matrix))

    for _ in tqdm(range(50000), desc="Generating additional split-brain cases"):
        nodes, matrix = generate_cluster()
        while not isSplitBrain(nodes, matrix):
            nodes, matrix = generate_cluster()
        processed = preprocess(nodes, matrix)
        for _ in range(3):
            x_train.append(processed)
            y_train.append(1)

    model.fit(x_train, y_train)
    end_time = time.time()

    print(f"Finished learning in {end_time - start_time:.2f} seconds")

    with open("split_brain_model_rf_2.pkl", "wb") as f:
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
    print("RF __________________")
    if isSingleType(nodes):
        return 0
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    return model.predict_proba(x_input)[0, 1]


def teach_rf(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    label = isSplitBrain(nodes, matrix)
    model.fit(x_input, [label])
    with open("split_brain_model_rf_2.pkl", "wb") as f:
        pickle.dump(model, f)
    return label
