import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Lambda, ELU, Dropout
from keras.layers.convolutional import Convolution2D

def generate_batch(images, steerings, batch_size=128):
  batch_images = np.zeros((batch_size, 160, 320, 3))
  batch_steering = np.zeros(batch_size)
  total = len(steerings)
  current = 0
  while 1:
    for i in range(batch_size):
      batch_images[i] = mpimg.imread('data/'+images[current])
      batch_steering[i] = steerings[current]
      current = (current + 1) % total
    yield batch_images, batch_steering

csv = pd.read_csv('data/driving_log.csv')
steerings = csv.steering.values
images = csv.center.values

X_train, X_valid, y_train, y_valid = train_test_split(images, steerings, test_size=0.1, random_state=1234)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
          input_shape=(160, 320, 3),
          output_shape=(160, 320, 3)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit_generator(generate_batch(X_train, y_train), verbose=1, samples_per_epoch=7296, nb_epoch=5,
  validation_data=generate_batch(X_valid, y_valid), nb_val_samples=896)

model.save_weights('model.h5')
with open('model.json', 'w') as f:
  f.write(model.to_json())