
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#%%
stock = pd.read_csv(r'C:\Users\Gireesh Sundaram\Downloads\NSEI.csv')

timesteps = stock["Date"].values
values = stock["Close"].values

#%%
plt.plot(timesteps, values)
plt.show()

#%%
window_size = 30

#%%
dataset = tf.data.Dataset.from_tensor_slices(values)
dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(1000)
dataset = dataset.batch(32).prefetch(1)

for x, y in dataset:
  print(x.numpy())
  print(y.numpy())

#%%
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(1)
])

"""
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
  lambda epochs: 1e-8 * 10 ** (epochs / 20)
)
"""

optimizer = tf.keras.optimizers.SGD(lr=0.0008, momentum=0.9)
model.compile(loss = 'mae', optimizer=optimizer, metrics=["mae"])
model.summary()
history = model.fit(dataset, epochs=30, verbose=2, callbacks= [lr_scheduler])
#history = model.fit(dataset, epochs=100, verbose=2, callbacks= [lr_scheduler])

#%%
plt.plot(history.history['lr'], history.history['loss'])
plt.show()

#%%
forecast = []

for time in range(len(values) - window_size):
  forecast.append(model.predict(values[time: time+window_size][np.newaxis]))

#%%
results = np.array(forecast)[:, 0, 0]

plt.plot(timesteps[window_size:], values[window_size:])
plt.plot(timesteps[window_size:], results)
plt.show()

#%%
tf.keras.metrics.mean_absolute_error(values[window_size:], results).numpy()
