import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load('imagenet_resized/8x8:0.1.0', with_info=True)
print(dataset)
print(info)
train_ds, valid_ds = dataset['train'], dataset['validation']
print(train_ds)
print(valid_ds)
info_feature = info.features
print(info_feature)
class_name = info_feature['label'].names
print(class_name[:5])
# for dic in train_ds.take(3):
#     plt.axis('off')
#     plt.title(dic['label'].numpy())
#     plt.imshow(dic['image'].numpy())
#     plt.show()
train_ds = train_ds.map(lambda dic: (dic['image'], dic['label']))
valid_ds = valid_ds.map(lambda dic: (dic['image'], dic['label']))
print(train_ds)

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1000, activation='softmax')
])
model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_ds.batch(32),
    validation_data=valid_ds.batch(32),
    epochs=5
)
print(model.summary())
print(history.history)
