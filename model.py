from keras.layers import Dense, Flatten, Lambda, PReLU, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

import util

# based on NVIDIA's paper
def get_model():
  model = Sequential()
  # normalization
  model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

  # convolutional and maxpooling layers
  model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), init='he_normal'))
  model.add(PReLU())
  model.add(BatchNormalization())

  model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), init='he_normal'))
  model.add(PReLU())
  model.add(BatchNormalization())

  model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), init='he_normal'))
  model.add(PReLU())
  model.add(BatchNormalization())

  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal'))
  model.add(PReLU())
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal'))
  model.add(PReLU())
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

  model.add(Flatten())

  # fully connected layers
  model.add(Dense(1164, init='he_normal'))
  model.add(PReLU())
  model.add(BatchNormalization())
  model.add(Dropout(0.8))
  
  model.add(Dense(100, init='he_normal'))
  model.add(PReLU())
  model.add(BatchNormalization())
  model.add(Dropout(0.8))
  
  model.add(Dense(50, init='he_normal'))
  model.add(PReLU())
  model.add(BatchNormalization())

  model.add(Dense(10, init='he_normal'))
  model.add(PReLU())
  model.add(BatchNormalization())

  model.add(Dense(1, activation='tanh'))

  return model

def main():
  # prepare data
  train, valid = util.prepare_data()
  
  # get model
  model = get_model()
  model.summary()

  # generators for training and validation
  BATCH = 128
  train_gen = util.next_train_batch(train, BATCH)
  valid_gen = util.next_valid_batch(valid, BATCH)

  # training
  EPOCHS = 5
  TRAINS = 20480
  VALIDS = 4096
  model.compile(optimizer=Adam(1e-2), loss="mse")
  history = model.fit_generator(train_gen,
                                samples_per_epoch=TRAINS,
                                nb_epoch=EPOCHS,
                                validation_data=valid_gen,
                                nb_val_samples=VALIDS,
                                verbose=1)

  # save model, weights
  model.save_weights('model.h5')
  with open('model.json', 'w') as f:
    f.write(model.to_json())

if __name__ == '__main__':
  main()