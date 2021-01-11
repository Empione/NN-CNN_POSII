import tensorflow as tf
import time

data_set = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data_set.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images / 255.0


num_epochs = 5


def model_build():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
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
print ('Test loss: {:.2f}, Test accuracy: {:.2f}'.format(test_loss, test_accuracy*100))


ex_model.save('Model_CNN.h5')