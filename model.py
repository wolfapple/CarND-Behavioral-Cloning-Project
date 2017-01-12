import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, Conv2D, merge, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def rgb_clahe(bgr_img,limit=5,grid=4):
  b,g,r = cv2.split(bgr_img)
  clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(grid,grid))
  b = clahe.apply(b)
  g = clahe.apply(g)
  r = clahe.apply(r)
  return cv2.merge([b,g,r])

def preprocess(img):
  roi = img[60:140, :, :]
  clahe = rgb_clahe(roi)
  resize = cv2.resize(clahe, (64, 64), interpolation=cv2.INTER_AREA)
  resize = (resize / 127.5) - 1.0
  return resize

def random_camera(row, angle=.25):
  camera = np.random.randint(3)
  if camera == 0:
    image = mpimg.imread('data/'+row.left.strip())
    steering = row.steering + angle
  elif camera == 1:
    image = mpimg.imread('data/'+row.center.strip())
    steering = row.steering
  else:
    image = mpimg.imread('data/'+row.right.strip())
    steering = row.steering - angle
  return image, steering

def random_flip(image, steering):
  if np.random.randint(2) == 0:
    image, steering = cv2.flip(image,1), -steering
  return image,steering

def random_translation(image, steering, x_range=100, y_range=10, angle=.4):
  rows, cols, _ = image.shape
  x = x_range * np.random.uniform() - x_range / 2
  steering = steering + (x / x_range * 2 * angle)
  y = y_range * np.random.uniform() - y_range / 2
  M = np.float32([[1,0,x], [0,1,y]])
  image = cv2.warpAffine(image, M, (cols, rows))
  return image, steering

def get_augmented_data(row):
  image, steering = random_camera(row)
  image, steering = random_flip(image, steering)
  image, steering = random_translation(image, steering)
  return preprocess(image), steering

def generate_train_batch(data, batch_size):
  images = np.zeros((batch_size, 64, 64, 3))
  steerings = np.zeros(batch_size)
  total = len(data)
  current = 0
  while True:
    for i in range(batch_size):
      row = data.iloc[current]
      image, steering = get_augmented_data(row)
      images[i] = image
      steerings[i] = steering
      current = (current + 1) % total
    yield images, steerings

def generate_valid_batch(data, batch_size):
  images = np.zeros((batch_size, 64, 64, 3))
  steerings = np.zeros(batch_size)
  total = len(data)
  current = 0
  while True:
    for i in range(batch_size):
      row = data.iloc[current]
      images[i] = preprocess(mpimg.imread('data/'+row.center.strip()))
      steerings[i] = row.steering
      current = (current + 1) % total
    yield images, steerings  

def read_csv(path):
  headers = ['center','left','right','steering','throttle','brake','speed']
  return pd.read_csv(path, names=headers, skiprows=1)

def prepare_data(data):
  data = data[data.speed > 15]
  train_nonzero = data[data.steering != 0]
  train_zero = (data[data.steering == 0]).sample(frac=.1)
  train = pd.concat([train_nonzero, train_zero], ignore_index=True)
  return shuffle(train)

def get_nvidia_model():
  model = Sequential()
  model.add(Convolution2D(24, 5, 5, init = 'he_normal', subsample= (2, 2), input_shape=(64, 64, 3)))
  model.add(ELU())
  model.add(Convolution2D(36, 5, 5, init = 'he_normal', subsample= (2, 2)))
  model.add(ELU())
  model.add(Convolution2D(48, 5, 5, init = 'he_normal', subsample= (2, 2)))
  model.add(ELU())
  model.add(Convolution2D(64, 3, 3, init = 'he_normal', subsample= (1, 1)))
  model.add(ELU())
  model.add(Convolution2D(64, 3, 3, init = 'he_normal', subsample= (1, 1)))
  model.add(ELU())
  model.add(Flatten())
  model.add(Dense(1164, init = 'he_normal'))
  model.add(ELU())
  model.add(Dense(100, init = 'he_normal'))
  model.add(ELU())
  model.add(Dense(50, init = 'he_normal'))
  model.add(ELU())
  model.add(Dense(10, init = 'he_normal'))
  model.add(ELU())
  model.add(Dense(1, init = 'he_normal'))
  return model

def get_comma_model():
  model = Sequential()
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(64, 64, 3)))
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

def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
    def f(X):
        div = X.get_shape()[axis].value // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)

def convolution2Dgroup(n_group, nb_filter, nb_row, nb_col, **kwargs):
    def f(input):
        return merge([
            Convolution2D(nb_filter//n_group,nb_row,nb_col, subsample=(1, 1), border_mode="same", activation="relu")(
                splittensor(axis=1,
                            ratio_split=n_group,
                            id_split=i)(input))
            for i in range(n_group)
        ],mode='concat',concat_axis=1)

    return f

def get_gtanet_model():
  inputs = Input(shape=(64,64,3))
  conv1 = Convolution2D(96, 11, 11, subsample=(4, 4), border_mode="same", activation="relu")(inputs)
  conv1 = BatchNormalization()(conv1)
  conv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid")(conv1)
  conv2 = convolution2Dgroup(2, 255, 5, 5)(conv1)
  conv2 = BatchNormalization()(conv2)
  conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid")(conv2)
  conv3 = Convolution2D(384, 3, 3, subsample=(1, 1), border_mode="same", activation="relu")(conv2)
  conv4 = convolution2Dgroup(2, 384, 3, 3)(conv3)
  conv5 = convolution2Dgroup(2, 256, 3, 3)(conv4)
  conv5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid")(conv5)
  flat = Flatten()(conv5)
  dense1 = Dense(4096, activation="relu")(flat)
  dense1 = Dropout(0.5)(dense1)
  dense2 = Dense(4096, activation="relu")(dense1)
  dense2 = Dropout(0.95)(dense2)
  output = Dense(1)(dense2)
  model = Model(input=inputs, output=output)
  return model

  # model = Sequential()
  # model.add(Convolution2D(96, 11, 11, subsample=(4, 4), border_mode="same", activation="relu", input_shape=(64, 64, 3))) # g1
  # model.add(BatchNormalization())
  # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid"))
  # model.add(Convolution2D(255, 5, 5, subsample=(1, 1), border_mode="same", activation="relu")) # g2
  # model.add(BatchNormalization())
  # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid"))
  # model.add(Convolution2D(384, 3, 3, subsample=(1, 1), border_mode="same", activation="relu")) # g1
  # model.add(Convolution2D(384, 3, 3, subsample=(1, 1), border_mode="same", activation="relu")) # g2
  # model.add(Convolution2D(256, 3, 3, subsample=(1, 1), border_mode="same", activation="relu")) # g2
  # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid"))
  # model.add(Flatten())
  # model.add(Dense(4096))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.5))
  # model.add(Dense(4096))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.95))
  # model.add(Dense(1))
  # return model


if __name__ == '__main__':
  # prepare data
  csv = read_csv('data/driving_log.csv')
  data = prepare_data(csv)

  # get model
  model = get_gtanet_model()
  model.summary()

  # training
  BATCH_SIZE = 128
  EPOCH = 5
  SAMPLES = BATCH_SIZE * 200
  NB_VAL_SAMPLES = round(len(data) / BATCH_SIZE) * BATCH_SIZE

  # lr = 2e-4
  model.compile(optimizer='adam', loss='mse')
  model.fit_generator(generate_train_batch(data, BATCH_SIZE), verbose=1, samples_per_epoch=SAMPLES, nb_epoch=EPOCH,
    validation_data=generate_valid_batch(data, BATCH_SIZE), nb_val_samples=NB_VAL_SAMPLES)

  # model save
  model.save_weights('model.h5')
  with open('model.json', 'w') as f:
    f.write(model.to_json())