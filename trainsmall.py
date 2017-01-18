from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout, Input
from keras.optimizers import SGD, Adam, Adamax, RMSprop
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from sklearn import metrics
import sys
import socket
import math
from keras.callbacks import Callback
import logging
import numpy as np
import math
from theano import tensor as T
from keras import backend as K
import csv
from pastalog import Log
from dataloader import data_generator, framewise_data_generator
from dataloader import framewise_valid_data


init = 'he_normal'
num_classes = 72
lr = 1e-3
optimiser = 'RMSprop'


class pastalog(Callback):

    def __init__(self):
        super(Callback, self).__init__()
        self.log_train = Log('http://localhost:5401',
                             'ants-challenge')

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs['val_loss']
        val_prediction_loss = logs['val_cls_prediction_loss']
        val_location_loss = logs['val_cls_location_loss']
        train_loss = logs['loss']
        prediction_loss = logs['cls_prediction_loss']
        location_loss = logs['cls_location_loss']
        self.log_train.post('train_loss', train_loss, epoch)
        self.log_train.post("cls_loss", prediction_loss, epoch)
        self.log_train.post("bbox_loss", location_loss, epoch)
        self.log_train.post("val_loss", val_loss, epoch)
        self.log_train.post("val_cls_loss", val_prediction_loss, epoch)
        self.log_train.post("val_bbox_loss", val_location_loss, epoch)


class LrReducer(Callback):

    def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=10, verbose=1):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_loss')
        lr = self.model.optimizer.lr.get_value()
        decay = math.pow(0.5, epoch / 50)
        new_lr = lr * decay
        new_lr = new_lr.astype('float32')
        self.model.optimizer.lr.set_value(new_lr)
        if current_score < self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print(
                    "--current best val_loss: {:.3f} in the epoch : {}".format(
                                current_score, epoch))
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= 10:
                    lr = self.model.optimizer.lr.get_value()
                    new_lr = lr * self.reduce_rate
                    print("Changing lr from {} to {}".format(lr, new_lr))
                    self.model.optimizer.lr.set_value(new_lr)
                    self.wait = 0
                    self.patience += 4
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            self.wait += 1


img_input = Input(shape=(3, 288, 512))

x = Convolution2D(96, 11, 11, subsample=(4, 4),
                  border_mode='same',
                  init=init, activation='relu')(img_input)

x = BatchNormalization(axis=1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Convolution2D(256, 5, 5, border_mode='same',
                  activation='relu', init=init)(x)
x = BatchNormalization(axis=1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Convolution2D(384, 3, 3, border_mode='same',
                  activation='relu', init=init)(x)
x = Convolution2D(384, 3, 3, border_mode='same',
                  activation='relu', init=init)(x)
x = Convolution2D(256, 3, 3, border_mode='same',
                  activation='relu', init=init)(x)
x = BatchNormalization(axis=1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


x = Flatten()(x)
x = Dense(4096, activation='relu', init=init)(x)
x = Dropout(0.5)(x)
x = Dense(1000, activation='relu', init=init)(x)
x = Dropout(0.5)(x)
cls_prediction = Dense(num_classes, activation='softmax', init=init,
                       name='cls_prediction')(x)
y = Dense(100, activation='relu', init=init, W_regularizer=l2(0.01))(x)
cls_location = Dense(num_classes*2, activation='tanh',
                     init=init, name='cls_location', W_regularizer=l2(0.01))(y)

model = Model(input=[img_input], output=[cls_prediction, cls_location])

if optimiser == 'SGD':
    optimisation_algo = SGD(
        lr=lr, decay=1e-6, momentum=0.9, nesterov=False)
elif optimiser == 'Adam':
    optimisation_algo = Adam(
        lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
elif optimiser == 'Adamax':
    optimisation_algo = Adamax(
        lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
elif optimiser == 'RMSprop':
    optimisation_algo = RMSprop(lr=lr, rho=0.9, epsilon=1e-08)
elif optimiser == 'Adadelta':
    optimisation_algo = Adadelta(lr=lr, rho=0.9, epsilon=1e-08)
elif optimiser == 'Adagrad':
    optimisation_algo = Adadelta(lr=lr, rho=0.9, epsilon=1e-08)

filename = 'Fixedcoord_AlexNet_lr_{:.2f}_{}'.format(lr, optimiser)
model_json_string = model.to_json()
open('/data_nas/ants/architectures/{}_model_architecture.json'.format(
    filename), 'w').write(model_json_string)
early_stop = EarlyStopping(
    monitor='val_loss', patience=10, verbose=0, mode='min')
checkpointer = ModelCheckpoint(filepath='/data_nas/ants/weights/' +
                               filename +
                               '_weights.{epoch:02d}-{val_loss:4.4f}.hdf5',
                               verbose=0, save_best_only=True,
                               monitor='val_loss', mode='min')


LR_reducer = LrReducer(patience=8, reduce_rate=0.5,
                       reduce_nb=10, verbose=0)

log = pastalog()

model.compile(optimizer=optimisation_algo,
              loss={'cls_prediction': 'binary_crossentropy',
                    'cls_location': 'mse'},
              loss_weights={'cls_prediction': 1., 'cls_location': 1})

model.fit_generator(framewise_data_generator(mode='train'),
                    samples_per_epoch=30000, nb_epoch=50,
                    validation_data=framewise_valid_data(nb_val_samples=3000),
                    callbacks=[checkpointer,
                    early_stop, log, LR_reducer])
