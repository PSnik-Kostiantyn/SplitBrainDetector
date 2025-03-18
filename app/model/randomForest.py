import os
import pickle
from sklearn.ensemble import RandomForestClassifier

from app.model.DataPreparation import generate_cluster, preprocess, isClusterDead, isSplitBrain, isSingleType


def train_model():
    print("Start learning")
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=42
    )

    # criterion='gini', max_features='sqrt'
    x_train, y_train = [], []

    for _ in range(500000):
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
    print("RF __________________")
    if (isSingleType(nodes)):
        return 0
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    return model.predict_proba(x_input)[0, 1]

def teach_rf(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    label = isSplitBrain(nodes, matrix)
    model.fit(x_input, [label])
    with open("split_brain_model_rf.pkl", "wb") as f:
        pickle.dump(model, f)
    return label
