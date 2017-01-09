import pandas as pd
import numpy as np
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

def generate_batch(data, batch_size=128):
  images = np.zeros((batch_size, 160, 320, 3))
  steerings = np.zeros(batch_size)
  total = len(data)
  current = 0
  while True:
    for i in range(batch_size):
      row = data.iloc[current]
      images[i] = mpimg.imread('data/'+row.center)
      steerings[i] = row.steering
      current = (current + 1) % total
    yield images, steerings

# read csv
headers = ['center','left','right','steering','throttle','brake','speed']
csv = pd.read_csv('data/driving_log.csv', names=headers, skiprows=1)

# split data
train_nonzero = csv[csv.steering != 0]
train_zero = (csv[csv.steering == 0]).sample(frac=.1)
train = pd.concat([train_nonzero, train_zero], ignore_index=True)
train_data, valid_data = train_test_split(shuffle(train), test_size=0.2, random_state=1234)

# model
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

model.fit_generator(generate_batch(train_data), verbose=1, samples_per_epoch=len(train_data), nb_epoch=5,
  validation_data=generate_batch(valid_data), nb_val_samples=len(valid_data))

model.save_weights('model.h5')
with open('model.json', 'w') as f:
  f.write(model.to_json())