import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Загрузка изображения и преобразование в нужный формат
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).resize((32, 32))
    image_array = np.array(image)
    binary_matrix = (image_array == [255, 255, 255]).all(axis=-1).astype(int)
    return binary_matrix.flatten()

# Softmax функция активации
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Для численной стабильности
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Кросс-энтропия
def multiclass_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Инициализация весов
def initialize_weights(input_size, num_classes):
    return np.random.randn(input_size, num_classes) * 0.01

# Прямое распространение
def forward_pass(inputs, weights):
    return softmax(np.dot(inputs, weights))

# Обновление весов
def update_weights(weights, inputs, delta, learning_rate):
    return weights + learning_rate * np.dot(inputs.T, delta)

# Обучение модели
def train_perceptron(X_train, y_train, num_classes, epochs=10, learning_rate=0.01):
    input_size = X_train.shape[1]
    weights = initialize_weights(input_size, num_classes)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X_train)):
            inputs = X_train[i].reshape(1, -1)
            true_output = y_train[i].reshape(1, -1)
            predicted_output = forward_pass(inputs, weights)
            delta = true_output - predicted_output
            weights = update_weights(weights, inputs, delta, learning_rate)
            loss = multiclass_cross_entropy_loss(true_output, predicted_output)
            epoch_loss += loss

        epoch_loss /= len(X_train)
        losses.append(epoch_loss)
        print(f"Эпоха {epoch+1}/{epochs}, Потери: {epoch_loss:.4f}")

    return weights, losses

# Тестирование модели
def test_perceptron(X_test, y_test, weights):
    correct_predictions = 0
    for i in range(len(X_test)):
        inputs = X_test[i].reshape(1, -1)
        true_output = y_test[i].reshape(1, -1)
        predicted_output = forward_pass(inputs, weights)
        if np.argmax(predicted_output) == np.argmax(true_output):
            correct_predictions += 1

    accuracy = correct_predictions / len(X_test)
    return accuracy

# Реализация one-hot encoding
def one_hot_encode(y, num_classes):
    encoded = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        encoded[i, label] = 1
    return encoded

# Разделение данных на тренировочные и тестовые
def train_test_split(X, y, test_size=0.2):
    dataset_size = len(X)
    test_size = int(dataset_size * test_size)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    X_train = X[indices[test_size:]]
    X_test = X[indices[:test_size]]
    y_train = y[indices[test_size:]]
    y_test = y[indices[:test_size]]
    return X_train, X_test, y_train, y_test

# Загрузка данных из CSV файла
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-2].values  # 1024 бита данных изображения
    y = data.iloc[:, -2].values   # Метка класса
    filenames = data.iloc[:, -1].values  # Имя файла изображения
    return X, y, filenames

# Преобразование меток в числовые значения
def label_to_numeric(y):
    classes = sorted(list(set(y)))
    label_map = {label: idx for idx, label in enumerate(classes)}
    y_numeric = np.array([label_map[label] for label in y])
    return y_numeric, len(classes), label_map

# Функция для построения графика потерь
def plot_loss_curve(losses):
    plt.plot(losses)
    plt.title("График изменения функции потерь")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери (Loss)")
    plt.show()

# Функция для работы с изображением пользователя
def test_user_image(image_path, weights, label_map):
    image_array = load_and_preprocess_image(image_path).reshape(1, -1)
    predicted_output = forward_pass(image_array, weights)
    probabilities = softmax(predicted_output)
    predicted_label_idx = np.argmax(probabilities)
    print("Вероятности для каждого класса:")
    for label, idx in label_map.items():
        print(f"{label}: {probabilities[0][idx] * 100:.2f}%")
    predicted_label = [label for label, idx in label_map.items() if idx == predicted_label_idx][0]
    print(f"\nРаспознанный класс изображения: {predicted_label}")

# GUI с использованием Tkinter
class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptron Trainer")
        self.canvas_size = 32
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Интерфейс
        self.epochs_label = tk.Label(root, text="Epochs:")
        self.epochs_label.pack()
        self.epochs_entry = tk.Entry(root)
        self.epochs_entry.insert(0, "10")
        self.epochs_entry.pack()

        self.learning_rate_label = tk.Label(root, text="Learning rate:")
        self.learning_rate_label.pack()
        self.learning_rate_entry = tk.Entry(root)
        self.learning_rate_entry.insert(0, "0.01")
        self.learning_rate_entry.pack()

        self.load_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_button.pack()

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack()

        self.canvas = tk.Canvas(root, width=self.canvas_size * 10, height=self.canvas_size * 10, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw_image)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.test_button = tk.Button(root, text="Test Image", command=self.test_image)
        self.test_button.pack()

    def load_data(self):
        file_path = filedialog.askopenfilename()
        self.X, self.y, _ = load_data(file_path)
        self.y_numeric, self.num_classes, self.label_map = label_to_numeric(self.y)
        self.y_one_hot = one_hot_encode(self.y_numeric, self.num_classes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_one_hot)

    def train_model(self):
        epochs = int(self.epochs_entry.get())
        learning_rate = float(self.learning_rate_entry.get())
        self.weights, self.losses = train_perceptron(self.X_train, self.y_train, self.num_classes, epochs, learning_rate)
        accuracy = test_perceptron(self.X_test, self.y_test, self.weights)
        messagebox.showinfo("Training Complete", f"Model Accuracy: {accuracy * 100:.2f}%")
        plot_loss_curve(self.losses)

    def draw_image(self, event):
        x, y = event.x // 10, event.y // 10
        self.canvas.create_rectangle(x * 10, y * 10, x * 10 + 10, y * 10 + 10, fill="black")
        self.draw.rectangle([x, y, x + 1, y + 1], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

    def test_image(self):
        image_array = np.array(self.image).flatten().reshape(1, -1)
        predicted_output = forward_pass(image_array, self.weights)
        probabilities = softmax(predicted_output)
        predicted_label_idx = np.argmax(probabilities)
        predicted_label = [label for label, idx in self.label_map.items() if idx == predicted_label_idx][0]

        # Формирование строки с вероятностями для каждого класса
        probabilities_text = "\n".join(
            f"{label}: {probabilities[0][idx] * 100:.2f}%" 
            for label, idx in self.label_map.items()
        )

        # Выводим результат и вероятности
        messagebox.showinfo("Test Result", f"Predicted class: {predicted_label}\n\nProbabilities:\n{probabilities_text}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()
