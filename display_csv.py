import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
import os

# Параметры
output_folder = 'output_images'  # Папка с изображениями
csv_filename = 'images_data_shuffled.csv'  # Имя CSV файла

# Чтение данных из CSV файла
images_data = []
with open(csv_filename, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        # Преобразуем строку в массив (первые 1024 бита)
        binary_matrix = np.array(row[:-2], dtype=int)  # Все кроме последних двух
        images_data.append((binary_matrix, row[-2], row[-1]))  # Класс и имя файла

# Извлечение изображений
images_to_display = images_data[1], images_data[12], images_data[561], images_data[31], images_data[112]

# Подготовка для отображения изображений
fig, axes = plt.subplots(2, len(images_to_display), figsize=(20, 4))

for i, (binary_matrix, class_name, image_id) in enumerate(images_to_display):
    # Преобразование в 32x32 пикселей
    image_array = binary_matrix.reshape(32, 32)
    
    # Отображение бинарного изображения
    axes[0, i].imshow(image_array, cmap='gray')
    axes[0, i].axis('off')
    
    # Открытие и отображение оригинального изображения
    original_image_path = os.path.join(output_folder, image_id)
    original_img = Image.open(original_image_path).convert("RGB")  # Открытие оригинального изображения
    axes[1, i].imshow(original_img)
    axes[1, i].axis('off')

# Подписи классов
for ax, (_, _, image_id) in zip(axes[0], images_to_display):
    ax.set_title(image_id)

plt.tight_layout()
plt.show()
