
import numpy as np
import csv
import matplotlib.pyplot as plt
import tensorflow as tf

#%%
def fig_plot(time, value):
    plt.plot(time, value)
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    plt.show()

#%%
timestep = []
close = []

with open(r"C:\Users\Gireesh Sundaram\Downloads\NSEI.csv") as nse:
    f = csv.reader(nse)
    next(f)
    for line in f:
        close.append(float(line[1]))
        timestep.append(len(close))

#%%
fig_plot(timestep, close)

#%%
split_time = 2250
time_train = timestep[:split_time]
close_train = close[:split_time]
time_test = timestep[split_time:]
close_test = close[split_time:]

#%%
window_size = 10
batch_size = 32
shuffle_buffer_size = 1000

#%%
def create_window_dataset(list):
    list = tf.expand_dims(list, axis = -1)
    ds = tf.data.Dataset.from_tensor_slices(list)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

#%%
train = create_window_dataset(time_train)

#%%
for x, y in train:
    print(x.numpy())
    print(y.numpy())

#%%
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters = 32, kernel_size=3, strides= 1, padding='causal',
                           activation = 'relu', input_shape=[None, 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20)
)

optimizer = tf.keras.optimizers.SGD()

model.compile(loss = 'mae', optimizer=optimizer, metrics=['mae'])
model.summary()
history = model.fit(train, epochs = 100, callbacks=[lr_scheduler], verbose=2)

#%%
plt.plot(history.history['lr'], history.history['loss'])
plt.axis([1e-8, 1e-3, 0, 1200])
plt.show()

#%%
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters = 32, kernel_size=3, strides= 1, padding='causal',
                           activation = 'relu', input_shape=[None, 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9)

model.compile(loss = 'mae', optimizer=optimizer, metrics=['mae'])
model.summary()
history = model.fit(train, epochs = 100, callbacks=[lr_scheduler], verbose=2)

#%%
