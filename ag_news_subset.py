import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load('ag_news_subset', with_info=True, as_supervised=True)
print(info)
# print(info.features['description'].names)
# print(info.features['label'].names)
# print(dataset)
class_name = info.features['label'].names
print(class_name)
train_ds, test_ds = dataset['train'], dataset['test']
print(train_ds)
print(test_ds)
# for e in train_ds.take(1):
#     for key, value in e:
#         print(key)
#         print(value)
# for feature, label in train_ds.take(5):
#     print(feature)
#     print(label)

Vectorize = tf.keras.layers.TextVectorization(
    max_tokens=1000,
    output_mode='int',
    output_sequence_length=100
)
train_text_ds = train_ds.map(lambda text, label: text).take(10)
Vectorize.adapt(train_text_ds)
vocab = Vectorize.get_vocabulary()
print(vocab)

model = tf.keras.models.Sequential([
    Vectorize,
    tf.keras.layers.Embedding(input_dim=1000, output_dim=128),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(4)
])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
history = model.fit(
    train_ds.batch(32),
    epochs=10
)
print(model.summary())
print(history.history)
evaluation = model.evaluate(
    test_ds.batch(32),
    verbose=2
)
print(evaluation)
model.save('ag_news_subset.keras')
