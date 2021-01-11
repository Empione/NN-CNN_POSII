import tensorflow as tf
import time


#Набор данных содержащий 70к изображений с 10-ю классами встроенный в Keras
data_set = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data_set.load_data()
#Нормализация
training_images  = training_images / 255.0
test_images = test_images / 255.0


#Params_for_model
num_neuron = 128
num_epochs = 15


def model_build():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_neuron, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])

    return model


ex_model = model_build()

start_t = time.time()
ex_model.fit(training_images, training_labels, epochs=num_epochs)
print("\nВремя обучения: {:.2f} mins".format((time.time() - start_t)/60))


test_loss, test_accuracy = ex_model.evaluate(test_images, test_labels)
print ('\nTest loss: {:.2f}, Test accuracy: {:.2f}'.format(test_loss, test_accuracy*100))


ex_model.save('Model_NN.h5')

