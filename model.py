from keras.layers import Dense, Flatten, Lambda, PReLU, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

import util

# based on NVIDIA's paper
def get_nvidia_model():
  model = Sequential()
  # normalization
  model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

  # convolutional and maxpooling layers
  model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), init='he_normal'))
  model.add(BatchNormalization())
  model.add(PReLU())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

  model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2), init='he_normal'))
  model.add(BatchNormalization())
  model.add(PReLU())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

  model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2), init='he_normal'))
  model.add(BatchNormalization())
  model.add(PReLU())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal'))
  model.add(BatchNormalization())
  model.add(PReLU())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal'))
  model.add(BatchNormalization())
  model.add(PReLU())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

  model.add(Flatten())

  # fully connected layers
  model.add(Dense(1164, init='he_normal'))
  model.add(BatchNormalization())
  model.add(PReLU())
  model.add(Dropout(0.8))

  model.add(Dense(100, init='he_normal'))
  model.add(BatchNormalization())
  model.add(PReLU())
  model.add(Dropout(0.8))

  model.add(Dense(50, init='he_normal'))
  model.add(BatchNormalization())
  model.add(PReLU())

  model.add(Dense(10, init='he_normal'))
  model.add(BatchNormalization())
  model.add(PReLU())

  model.add(Dense(1, init='he_normal'))

  return model

def main():
  # get model
  model = get_nvidia_model()
  model.summary()

  # generators for training and validation
  BATCH = 64
  train_gen = util.next_train_batch(BATCH)
  valid_gen = util.next_valid_batch(BATCH)

  # training
  EPOCHS = 10
  TRAINS = 20480
  VALIDS = 6400
  model.compile(optimizer=Adam(1e-4), loss="mse")
  history = model.fit_generator(train_gen,
                                samples_per_epoch=TRAINS,
                                nb_epoch=EPOCHS,
                                validation_data=valid_gen,
                                nb_val_samples=VALIDS,
                                verbose=1,
                                callbacks=[
                                  ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True),
                                  EarlyStopping(patience=3, verbose=1)
                                ])

  # save model, weights
  with open('model.json', 'w') as f:
    f.write(model.to_json())

if __name__ == '__main__':
  main()