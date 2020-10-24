#Подключаем библиотеки
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import VisualPredictions as vs

class_names = ['Audi', 'BMW', 'Škoda', 'Mercedes-Benz']

base_folder = "D:/Программирование/Project Python/PyMachineLearningCars/img_train"

train_folder_audi = base_folder + "/audi/*.jpg";
train_folder_bmw = base_folder + "/bmw/*.jpg";
train_folder_skoda = base_folder + "/skoda/*.jpg";
train_folder_mercedes_benz = base_folder + "/mercedes_benz/*.jpg";

test_folder_audi = base_folder + "/test_audi/*.jpg";
test_folder_bmw = base_folder + "/test_bmw/*.jpg";
test_folder_skoda = base_folder + "/test_skoda/*.jpg";
test_folder_mercedes_benz = base_folder + "/test_mercedes_benz/*.jpg";
#---------Загрузка изображений------------------------------------------------------------------------------
s_img_w = 280
s_img_h = 200

train_images_audi = vs.load_images_from_folder(train_folder_audi, s_img_w, s_img_h)
train_images_bmw = vs.load_images_from_folder(train_folder_bmw, s_img_w, s_img_h)
train_images_skoda = vs.load_images_from_folder(train_folder_skoda, s_img_w, s_img_h)
train_images_mercedes_benz = vs.load_images_from_folder(train_folder_mercedes_benz, s_img_w, s_img_h)

test_images_audi = vs.load_images_from_folder(test_folder_audi, s_img_w, s_img_h)
test_images_bmw = vs.load_images_from_folder(test_folder_bmw, s_img_w, s_img_h)
test_images_skoda = vs.load_images_from_folder(test_folder_skoda, s_img_w, s_img_h)
test_images_mercedes_benz = vs.load_images_from_folder(test_folder_mercedes_benz, s_img_w, s_img_h)

#---------Формирование списков------------------------------------------------------------------------------
train_images = np.array(list(train_images_audi) + list(train_images_bmw)+ list(train_images_skoda)+ list(train_images_mercedes_benz))
train_labels = np.array([0]*len(train_images_audi) + [1]*len(train_images_bmw) + [2]*len(train_images_skoda) +[3]*len(train_images_mercedes_benz))

test_images = np.array(list(test_images_audi) + list(test_images_bmw)+ list(test_images_skoda)+ list(test_images_mercedes_benz))
test_labels = np.array([0]*len(test_images_audi) + [1]*len(test_images_bmw)+ [2]*len(test_images_skoda)+[3]*len(test_images_mercedes_benz))


print("Count train images all: {}".format(len(train_images)))
print("Count train labels all: {}".format(len(train_labels)))

print("Count test images all: {}".format(len(test_images)))

#1- нормализация 0-255→0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

#plt.figure()
#plt.imshow(test_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#plt.figure(figsize=(10,10))
#for i in range(len(test_images)):
#    plt.subplot(4,4,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(test_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[test_labels[i]])
#plt.show()

# 2- Построение модели-------------------------------------------------------

# Создаём последовательную модель
model = keras.Sequential()
# Добавляем слои

# 1 - слой в сети tf.keras.layers.Flatten 
# преобразует формат изображений из 2d-массива (s_img_w на s_img_hпикселей) в 1d-массив из s_img_w * s_img_h= N пикселей
model.add(keras.layers.Flatten(input_shape=(s_img_w, s_img_h, 3)))

# 2 -слой Dense содержит N узлов (или нейронов).
model.add(keras.layers.Dense(200, activation=tf.nn.elu))

#3 -слой Dense содержит N узлов (или нейронов).
model.add(keras.layers.Dense(200, activation=tf.nn.relu))

# 3 (и последний) уровень — это слой с N узлами tf.nn.softmax, 
# который возвращает массив из len(class_names) вероятностных оценок, сумма которых равна 1
model.add(keras.layers.Dense(len(class_names), activation=tf.nn.softmax))
# End Построение модели----------------------------------------

# Скомпилирование модели--------------------------
model.compile(optimizer=tf.optimizers.Adam(), # Optimizer (оптимизатор) — это то, как модель обновляется на основе данных, которые она видит
              loss='sparse_categorical_crossentropy', # Loss function (функция потери) — измеряет насколько точная модель во время 
              metrics=['accuracy']) # Metrics (метрики) — используется для контроля за этапами обучения и тестирования

# Обучение модели

# Подача данных обучения модели (в этом примере — массивы train_images и train_labels)
# Модель учится ассоциировать изображения и метки
# При моделировании модели отображаются показатели потерь (loss) и точности (acc).
model.fit(train_images, train_labels, epochs=7)

# Оценка точности
# Мы просим модель сделать прогнозы о тестовом наборе (в этом примере — массив test_images). 
# Мы проверяем соответствие прогнозов меток из массива меток (в этом примере — массив test_labels)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Прогнозирование
predictions = model.predict(test_images)

#Предсказание представляет собой массив из N чисел. 
#Они описывают «уверенность» модели в том, что изображение соответствует каждому из N разных классов
print(predictions[0])

num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  vs.plot_image(plt,i, predictions, test_labels, test_images, class_names)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  vs.plot_value_array(plt,i, predictions, test_labels, len(class_names))
plt.show()