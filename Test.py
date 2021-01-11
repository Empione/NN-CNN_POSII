import tensorflow as tf
import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image_path = '4_Coat.jpg'

num_model = 0
img_size = (28, 28)
img_size_flat = np.prod(img_size)


def img_prep(image_path, i):
    img = Image.open(image_path).convert('L').resize((28, 28))
    inverted_image = PIL.ImageOps.invert(img)
    if i == 0:
        arr_image = np.array(inverted_image).reshape(1, 28, 28) / 255.0
        print(arr_image)
    else:
        arr_image = np.array(inverted_image).reshape(1, np.prod(img_size)) / 255.0
        print(arr_image)
    plt.figure()
    plt.imshow(inverted_image)
    plt.colorbar()
    plt.show()
    return arr_image


arr = ['Model_NN.h5', 'Model_CNN.h5']
test_model = tf.keras.models.load_model(arr[num_model])

load_image = img_prep(image_path, num_model)
predictions = test_model.predict(load_image)
print('\n{}'.format(np.argmax(predictions)))