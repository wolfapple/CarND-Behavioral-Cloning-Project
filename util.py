import errno
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

def roi(image, top=60, bottom=140):
  return image[top:bottom, :]

def resize(image, dim=(64, 64)):
  return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

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

def get_augmented_image(image, steering):
  if np.random.binomial(1, 0.9):
    image, steering = random_shear(image, steering)
  image, steering = random_flip(image, steering)
  image = adjust_gamma(image)
  image = roi(image)
  image = resize(image)
  return image, steering

def get_next_image_files(batch_size=64):
  STEERING_COEFFICIENT = 0.229
  data = pd.read_csv('data/driving_log.csv')
  num_of_img = len(data)
  rnd_indices = np.random.randint(0, num_of_img, batch_size)

  image_files_and_angles = []
  for index in rnd_indices:
    rnd_image = np.random.randint(0, 3)
    if rnd_image == 0:
      img = data.iloc[index]['left'].strip()
      angle = data.iloc[index]['steering'] + STEERING_COEFFICIENT
      image_files_and_angles.append((img, angle))

    elif rnd_image == 1:
      img = data.iloc[index]['center'].strip()
      angle = data.iloc[index]['steering']
      image_files_and_angles.append((img, angle))
    else:
      img = data.iloc[index]['right'].strip()
      angle = data.iloc[index]['steering'] - STEERING_COEFFICIENT
      image_files_and_angles.append((img, angle))

  return image_files_and_angles

def next_batch(batch_size=64):
  while True:
    images = []
    steerings = []
    images = get_next_image_files(batch_size)
    for img_file, angle in images:
      raw_image = mpimg.imread('data/' + img_file)
      raw_angle = angle
      new_image, new_angle = get_augmented_image(raw_image, raw_angle)
      images.append(new_image)
      steerings.append(new_angle)

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