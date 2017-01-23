from keras.layers import Dense, Flatten, Lambda, PReLU, MaxPooling2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

import util
import math
import time

# based on NVIDIA's paper
def get_model(number):
  if number == 0:
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

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

    model.add(Dense(1164, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(100, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(50, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())

    model.add(Dense(10, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())

    model.add(Dense(1, activation='tanh'))

  elif number == 1:
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('relu'))
    
    model.add(Dense(100))
    model.add(Activation('relu'))
    
    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

  return model

def main():
  # for n in range(1):
    n = 0
    train, valid = util.prepare_data()

    # get model
    model = get_model(n)

    # generators for training and validation
    BATCH = 128
    train_gen = util.next_train_batch(train, BATCH)
    valid_gen = util.next_valid_batch(valid, BATCH)

    # training
    EPOCHS = 5
    TRAINS = 20480
    VALIDS = 4096
    model.compile(optimizer=Adam(2e-3), loss="mse")
    start = time.time()
    history = model.fit_generator(train_gen,
                                  samples_per_epoch=TRAINS,
                                  nb_epoch=EPOCHS,
                                  validation_data=valid_gen,
                                  nb_val_samples=VALIDS,
                                  verbose=1,
                                  callbacks=[
                                    # ModelCheckpoint('model'+str(n)+'.h5', verbose=1, save_best_only=True, save_weights_only=True),
                                    # EarlyStopping(patience=3, verbose=1)
                                  ])

    print('Model #'+str(n)+' time: ' + str(time.time() - start))
    print('Model #'+str(n)+' Best val_loss: ' + str(min(history.history['val_loss'])))

    # save model, weights
    model.save_weights('model'+str(n)+'.h5')
    with open('model'+str(n)+'.json', 'w') as f:
      f.write(model.to_json())

if __name__ == '__main__':
  main()