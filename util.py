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

def random_camera(row, angle=0.229):
  camera = np.random.randint(0, 3)
  if camera == 0:
    image_path = row.left.strip()
    steering = row.steering + angle
  elif camera == 1:
    image_path = row.center.strip()
    steering = row.steering
  else:
    image_path = row.right.strip()
    steering = row.steering - angle

  return mpimg.imread('data/' + image_path), steering

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

def random_shear(image, steering, shear_range=200):
  rows, cols, _ = image.shape
  dx = np.random.randint(-shear_range, shear_range + 1)
  random_point = [cols / 2 + dx, rows / 2]
  pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
  pts2 = np.float32([[0, rows], [cols, rows], random_point])
  dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
  M = cv2.getAffineTransform(pts1, pts2)
  image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)  
  return image, steering + dsteering

def random_bumpy(image, y_range=15):
  rows, cols, _ = image.shape
  dy = (y_range * np.random.uniform()) - (y_range / 2)
  M = np.float32([[1, 0, 0], [0, 1, dy]])
  return cv2.warpAffine(image, M, (cols, rows))

def get_augmented_data(row):
  image, steering = random_camera(row)
  if np.random.binomial(1, 0.9):
    image, steering = random_shear(image, steering)
  image, steering = random_flip(image, steering)
  image = adjust_gamma(image)
  image = preprocess(image)
  image = random_bumpy(image)
  return image, steering

def next_batch(batch_size):
  data = pd.read_csv('data/driving_log.csv')
  total = len(data)
  while True:
    images = []
    steerings = []
    random_indices = np.random.randint(0, total, batch_size)
    for idx in random_indices:
      row = data.iloc[idx]
      new_image, new_steering = get_augmented_data(row)
      images.append(new_image)
      steerings.append(new_steering)

    yield np.array(images), np.array(steerings)

def save_model(model):
  safe_delete('model.json')
  safe_delete('model.h5')

  with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())

  model.save_weights('model.h5')

def safe_delete(file):
  if os.path.exists(file):
    os.remove(file)
