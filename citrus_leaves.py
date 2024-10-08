import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


dataset, info = tfds.load('citrus_leaves', with_info=True, as_supervised=True)
print(dataset)
print(info)
ds = dataset['train']
print(ds)
class_name = info.features['label'].names
print(class_name)
# for image, label in ds.take(5):
#     plt.axis('off')
#     plt.imshow(image.numpy())
#     plt.title(class_name[label.numpy()])
#     plt.show()
# ds_length = ds.cardinality().numpy()
# print(ds_length)


model = tf.keras.models.Sequential([
    tf.keras.layers.Resizing(256, 256),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
history = model.fit(
    ds.batch(32),
    epochs=10,
)
print(history.history)
print(model.summary())

model.save('citrus_leaves.keras')
