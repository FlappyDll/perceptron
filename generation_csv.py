import os
import random
import numpy as np
from PIL import Image
import csv

# Параметры
input_folder = 'emojis'  # Папка с изображениями
output_folder = 'output_images'  # Папка для сохранения новых изображений
classes = ['Smile', 'Sad', 'Neutral', 'Smirk', 'Interesting', 'Crying', 'Silent', 'Aggressive', 'Unbelievable', 'The_end']
num_images_per_class = 11
num_rotations = 13
image_size = (32, 32)
unique_id = -1

# Создание выходных папок
os.makedirs(output_folder, exist_ok=True)

# Функция для генерации рандомных поворотов и создания новых изображений
def generate_rotated_images(unique_id):
    images_data = []

    for class_idx in range(1, 11):  # 10 классов
        class_name = classes[class_idx - 1]

        for image_idx in range(1, num_images_per_class + 1):
            image_name = f'{class_idx:01d}-{image_idx:02d}.bmp'
            image_path = os.path.join(input_folder, image_name)

            # Чтение исходного изображения
            img = Image.open(image_path).resize(image_size)

            for rotation_idx in range(num_rotations):
                # Случайный угол поворота от -20 до +20 градусов
                angle = [-20, -18, -15, -10, -7, -5, 0, 5, 7, 10, 15, 18, 20]
                # Поворот изображения
                rotated_img = img.rotate(angle[rotation_idx], expand=False, fillcolor="white")
                #rotated_img = rotated_img.resize(image_size)  # Изменение размера обратно на 32x32

                # Генерация уникального идентификатора
                unique_id = unique_id+1

                # Сохранение нового изображения
                new_image_id = f'{class_name}_{unique_id}.bmp'
                rotated_image_path = os.path.join(output_folder, new_image_id)
                rotated_img.save(rotated_image_path)

                # Преобразование изображения в массив 1 и 0 (белые пиксели = 1, остальные = 0)
                image_array = np.array(rotated_img)
                binary_matrix = (image_array == [255, 255, 255]).all(axis=-1).astype(int)

                # Добавление данных для CSV файла
                images_data.append((binary_matrix.flatten(), class_name, new_image_id))

    return images_data

# Функция для записи данных в CSV файл
def write_to_csv(images_data, csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for binary_matrix, class_name, image_id in images_data:
            writer.writerow(np.concatenate([binary_matrix, [class_name, image_id]]))

# Генерация изображений и данных для CSV
images_data = generate_rotated_images(unique_id)

# Запись оригинального CSV файла
write_to_csv(images_data, 'images_data.csv')

# Перемешивание строк и запись во второй CSV файл
random.shuffle(images_data)
write_to_csv(images_data, 'images_data_shuffled.csv')

print("Генерация изображений и CSV файлов завершена.")
