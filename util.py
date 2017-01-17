import errno
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

def roi(image, top, bottom):
  return image[top:bottom, :]

def resize(image, dim):
  return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def preprocess(image, top=60, bottom=140, dim=(64, 64)):
  return resize(roi(image, top, bottom), dim)

def random_camera(row, angle):
  camera = np.random.randint(0, 3)
  if camera == 0:
    image = mpimg.imread('data/' + row.left.strip())
    steering = row.steering + angle
  elif camera == 1:
    image = mpimg.imread('data/' + row.center.strip())
    steering = row.steering
  else:
    image = mpimg.imread('data/' + row.right.strip())
    steering = row.steering - angle

  return image, steering

def random_flip(image, steering):
  if np.random.binomial(1, 0.5):
    return cv2.flip(image, 1), -steering
  else:
    return image, steering

def adjust_gamma(image):
  gamma = np.random.uniform(0.4, 1.5)
  inv_gamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** inv_gamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(image, table)

def random_shear(image, steering, shear_range):
  rows, cols, _ = image.shape
  dx = np.random.randint(-shear_range, shear_range + 1)
  random_point = [cols / 2 + dx, rows / 2]
  pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
  pts2 = np.float32([[0, rows], [cols, rows], random_point])
  dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
  M = cv2.getAffineTransform(pts1, pts2)
  image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)  
  return image, steering + dsteering

def random_bumpy(image, y_range=40):
  rows, cols, _ = image.shape
  dy = (y_range * np.random.uniform()) - (y_range / 2)
  M = np.float32([[1, 0, 0], [0, 1, dy]])
  return cv2.warpAffine(image, M, (cols, rows))

def get_augmented_data(row):
  image, steering = random_camera(row, angle=0.229)
  if np.random.binomial(1, 0.9):
    image, steering = random_shear(image, steering, shear_range=200)
  image, steering = random_flip(image, steering)
  image = adjust_gamma(image)
  # image = random_bumpy(image)
  return image, steering

def next_batch(batch_size=64):
  data = pd.read_csv('data/driving_log.csv')
  while True:
    images = []
    steerings = []
    random_indices = np.random.randint(0, len(data), batch_size)
    for idx in random_indices:
      row = data.iloc[idx]
      new_image, new_steering = get_augmented_data(row)
      images.append(preprocess(new_image))
      steerings.append(new_steering)

    yield np.array(images), np.array(steerings)

def save_model(model, model_name='model.json', weights_name='model.h5'):
  silent_delete(model_name)
  silent_delete(weights_name)
  
  with open(model_name, 'w') as outfile:
    outfile.write(model.to_json())

  model.save_weights(weights_name)

def silent_delete(file):
  try:
    os.remove(file)

  except OSError as error:
    if error.errno != errno.ENOENT:
      raise