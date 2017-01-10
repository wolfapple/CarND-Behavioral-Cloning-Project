import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

def rgb_clahe(bgr_img,limit=3,grid=4):
    b,g,r = cv2.split(bgr_img)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(grid,grid))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    return cv2.merge([b,g,r])

def preprocess(img):
  roi = img[70:130, :, :]
  resize = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
  clahe = rgb_clahe(resize)
  return clahe

def generate_batch(data, batch_size=128):
  images = np.zeros((batch_size, 64, 64, 3))
  steerings = np.zeros(batch_size)
  total = len(data)
  current = 0
  while True:
    for i in range(batch_size):
      row = data.iloc[current]
      images[i] = preprocess(mpimg.imread('data/'+row.center))
      steerings[i] = row.steering
      current = (current + 1) % total
    yield images, steerings

def read_csv(path):
  headers = ['center','left','right','steering','throttle','brake','speed']
  return pd.read_csv(path, names=headers, skiprows=1)

def split_data(data):
  train_nonzero = csv[csv.steering != 0]
  train_zero = (csv[csv.steering == 0]).sample(frac=.1)
  train = pd.concat([train_nonzero, train_zero], ignore_index=True)
  return train_test_split(shuffle(train), test_size=0.2, random_state=1234)

def get_model():
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(64, 64, 3),
            output_shape=(64, 64, 3)))
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
  return model

if __name__ == '__main__':
  # prepare data
  csv = read_csv('data/driving_log.csv')
  train_data, valid_data = split_data(csv)

  # get model
  model = get_model()
  model.summary()

  # training
  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  model.fit_generator(generate_batch(train_data), verbose=1, samples_per_epoch=3328, nb_epoch=5,
    validation_data=generate_batch(valid_data), nb_val_samples=896)

  # model save
  model.save_weights('model.h5')
  with open('model.json', 'w') as f:
    f.write(model.to_json())