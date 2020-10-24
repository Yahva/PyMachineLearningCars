import numpy as np
import glob
from PIL import Image

# метод для вывода изображения с процентом соответствия категории
def plot_image(plt, i, predictions_array, true_label, img, class_names):
  prediction, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(prediction)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(plt, i, predictions_array, true_label, count_class):
  prediction, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(count_class), prediction, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(prediction)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


def load_images_from_folder(folder, s_w, s_h):
    images = []
    for f in glob.iglob(folder):
        img = Image.open(f) 
        img = img.resize((s_w, s_h), Image.ANTIALIAS) 
        images.append(np.asarray(img))
    images = np.array(images)
    return images


