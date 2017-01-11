import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D

def rgb_clahe(bgr_img,limit=3,grid=4):
    b,g,r = cv2.split(bgr_img)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(grid,grid))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    return cv2.merge([b,g,r])

def preprocess(img):
  roi = img[65:135, :, :]
  clahe = rgb_clahe(roi)
  resize = cv2.resize(clahe, (64, 64), interpolation=cv2.INTER_AREA)
  resize = (resize / 127.5) - 1.0
  return resize

def random_camera(row, angle=.2):
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

def random_translation(image, steering, x_range=100, y_range=20, angle=.3):
  rows, cols, _ = image.shape
  x = x_range * np.random.uniform() - x_range / 2
  steering = steering + (x / x_range * 2 * angle)
  y = y_range * np.random.uniform() - y_range / 2
  M = np.float32([[1,0,x], [0,1,y]])
  image = cv2.warpAffine(image, M, (cols, rows))
  return image, steering

def random_rotation(image, steering, ang_range=10):
  ang_rot = np.random.uniform(ang_range) - ang_range / 2
  rows, cols, _ = img.shape
  M = cv2.getRotationMatrix2D((cols/2,rows/2), ang_rot, 1)
  image = cv2.warpAffine(image, M, (cols,rows))
  steering = steering + 0
  return image,steering

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
  data = data[data.throttle != 0]
  return shuffle(data)

def get_model():
  model = Sequential()
  model.add(Convolution2D(3,1,1,  border_mode='valid', name='conv0', init='he_normal', input_shape=(64,64,3)))
  model.add(Convolution2D(32,3,3, border_mode='valid', name='conv1', init='he_normal'))
  model.add(ELU())
  model.add(Convolution2D(32,3,3, border_mode='valid', name='conv2', init='he_normal'))
  model.add(ELU())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.5))
  
  model.add(Convolution2D(64,3,3, border_mode='valid', name='conv3', init='he_normal'))
  model.add(ELU())
  model.add(Convolution2D(64,3,3, border_mode='valid', name='conv4', init='he_normal'))
  model.add(ELU())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.5))
  
  model.add(Convolution2D(128,3,3, border_mode='valid', name='conv5', init='he_normal'))
  model.add(ELU())
  model.add(Convolution2D(128,3,3, border_mode='valid', name='conv6', init='he_normal'))
  model.add(ELU())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.5))
  
  model.add(Flatten())
  model.add(Dense(512,name='hidden1', init='he_normal'))
  model.add(ELU())
  model.add(Dropout(0.5))
  model.add(Dense(64,name='hidden2', init='he_normal'))
  model.add(ELU())
  model.add(Dropout(0.5))
  model.add(Dense(16,name='hidden3',init='he_normal'))
  model.add(ELU())
  model.add(Dropout(0.5))
  model.add(Dense(1, name='output', init='he_normal'))  
  return model

if __name__ == '__main__':
  # prepare data
  csv = read_csv('data/driving_log.csv')
  data = prepare_data(csv)

  # get model
  model = get_model()
  model.summary()

  # training
  BATCH_SIZE = 128
  EPOCH = 10
  SAMPLES = BATCH_SIZE * 200
  NB_VAL_SAMPLES = round(len(data) / BATCH_SIZE) * BATCH_SIZE

  model.compile(optimizer='adam', loss='mse')
  model.fit_generator(generate_train_batch(data, BATCH_SIZE), verbose=1, samples_per_epoch=SAMPLES, nb_epoch=EPOCH,
    validation_data=generate_valid_batch(data, BATCH_SIZE), nb_val_samples=NB_VAL_SAMPLES)

  # model save
  model.save_weights('model.h5')
  with open('model.json', 'w') as f:
    f.write(model.to_json())