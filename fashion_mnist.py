import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# a task of classification
# (image, label)

mnist_dataset, mnist_info = tfds.load('mnist', with_info=True, as_supervised=True)
print(mnist_dataset)
print(mnist_info)
train_ds, test_ds = mnist_dataset['train'], mnist_dataset['test']
print(train_ds)
print(test_ds)
# we have get train_ds and test_ds
# our task is to classify image of (28, 28, 1) to get the number in it

# we can try to show some (image, label) in train_ds
for image, label in train_ds.take(1):
    plt.axis('off')
    plt.imshow(image.numpy())
    plt.title(label.numpy())
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.Conv2D(32, (2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10),
])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
history = model.fit(
    train_ds.batch(32).cache(), epochs=1
)
# show model
print(model.summary())
# show the loss and accuracy in every epoch in training
# print(history.history)
train_loss, train_accuracy = history.history['loss'], history.history['sparse_categorical_accuracy']
print(f'train_ds loss: {train_loss}')
print(f'train_ds accuracy: {train_accuracy}')
# show the loss, accuracy in test
test_loss, test_accuracy = model.evaluate(test_ds.batch(32).cache(), verbose=2)
print(f'test_ds loss: {test_loss}, accuracy loss: {test_accuracy}')

# So we have completed the model that can classify the number in image

# At last, we can choose to save the model and get a file in keras format
model.save('fashion_mnist.keras')
