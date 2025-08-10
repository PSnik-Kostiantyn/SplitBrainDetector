import os
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from app.model.DataPreparation import *


def build_neural_network(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model():
    max_nodes = 15
    input_dim = max_nodes + (max_nodes * max_nodes)
    model = build_neural_network(input_dim)

    scaler = StandardScaler()

    X_train, y_train = [], []

    for _ in tqdm(range(500000), desc="Generating normal data"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X_train.append(preprocess(nodes, matrix))
        y_train.append(isSplitBrain(nodes, matrix))

    for _ in tqdm(range(300000), desc="Generating additional split-brain cases"):
        nodes, matrix = generate_cluster()
        while not isSplitBrain(nodes, matrix):
            nodes, matrix = generate_cluster()
        sample = preprocess(nodes, matrix)
        for _ in range(2):
            X_train.append(sample)
            y_train.append(1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = scaler.fit_transform(X_train)

    print("Training neural network...")
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1, callbacks=[early_stopping])

    model.save("split_brain_model_nn.keras")

    import pickle
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model


def load_nn_model():
    model_path = "split_brain_model_nn.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    else:
        return train_model()


def predict_nn(nodes, matrix):
    print("NN __________________")
    if isSingleType(nodes):
        return 0
    model = load_nn_model()

    scaler_path = "scaler.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler()

    x_input = preprocess(nodes, matrix).reshape(1, -1)
    x_input = scaler.transform(x_input)
    return model.predict(x_input, verbose=0)[0, 0]


def teach_nn(nodes, matrix):
    model = load_nn_model()

    # Завантаження scaler
    scaler_path = "scaler.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler()

    x_input = preprocess(nodes, matrix).reshape(1, -1)
    x_input = scaler.transform(x_input)
    label = isSplitBrain(nodes, matrix)

    other_label = 1 - label
    dummy_input = np.zeros_like(x_input)

    X = np.vstack([x_input, dummy_input])
    y = np.array([label, other_label])

    model.fit(X, y, epochs=1, batch_size=2, verbose=0)

    proba = model.predict(x_input, verbose=0)
    current_loss = log_loss([label], proba, labels=[0, 1])

    model.save("split_brain_model_nn.keras")
    return current_loss