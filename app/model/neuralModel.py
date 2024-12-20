import os
import random
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


from app.model.isDeadCluster import isClusterDead
from app.model.isSplitBrain import isSplitBrain


# Генерація одного набору даних
def generate_data():
    while True:
        num_nodes = random.randint(2, 9)
        nodes = [random.choice(['A', 'B', 'C']) + str(i) for i in range(num_nodes)]
        matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes))
        np.fill_diagonal(matrix, 0)

        if isClusterDead(nodes, matrix):
            continue

        split_brain = isSplitBrain(nodes, matrix)
        return nodes, matrix, int(split_brain)


# Генерація всіх даних
def generate_dataset(samples=70000):
    X, y = [], []
    for _ in range(samples):
        nodes, matrix, split_brain = generate_data()
        flattened_matrix = matrix.flatten().tolist()
        X.append(flattened_matrix)
        y.append(split_brain)
    return np.array(X, dtype=object), np.array(y)


# Функція передбачення
def predict_neural_model(nodes, matrix):
    model_path = 'split_brain_model.h5'

    # Завантаження або створення моделі
    if not os.path.exists(model_path):
        print("Start learning")
        X, y = generate_dataset(70000)

        # Визначення фіксованого розміру матриці для нейронної мережі
        max_matrix_size = max(len(row) for row in X)
        X = np.array([np.pad(row, (0, max_matrix_size - len(row)), constant_values=0) for row in X], dtype=float)

        model = Sequential([
            Dense(128, input_dim=X.shape[1], activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
        model.save(model_path)
        print("Model saved")
    else:
        model = load_model(model_path)

    # Передбачення
    num_nodes = len(nodes)
    flattened_matrix = np.array(matrix).flatten()
    padded_matrix = np.pad(flattened_matrix, (0, num_nodes ** 2 - len(flattened_matrix)), constant_values=0)
    input_data = padded_matrix.reshape(1, -1)
    prediction = model.predict(input_data)[0][0]
    return round(prediction * 100, 2)