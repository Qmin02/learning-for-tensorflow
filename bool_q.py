import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

ds_list = tfds.list_builders()
print(True if 'bool_q' in ds_list else False)
# for e in ds_list:
#     if e[0] == 'b':
#         print(e)

dataset, info = tfds.load('bool_q', with_info=True)
print(dataset)
print(info)
train_ds, valid_ds = dataset['train'], dataset['validation']
print(train_ds)
print(valid_ds)
train_ds = train_ds.map(lambda e: (e['question'], tf.cast(e['answer'], tf.int32)))
valid_ds = valid_ds.map(lambda e: (e['question'], tf.cast(e['answer'], tf.int32)))
print(train_ds)
print(valid_ds)

TextLine = tf.keras.layers.TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=10)
TextLine.adapt(train_ds.map(lambda text, label: text))
vocab = TextLine.get_vocabulary()
print(len(vocab))
model = tf.keras.models.Sequential([
    TextLine,
    tf.keras.layers.Embedding(input_dim=1000, output_dim=128),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
history = model.fit(
    train_ds.batch(32),
    epochs=5,
    validation_data=valid_ds.batch(32)
)
print(model.summary())
print(history.history)
