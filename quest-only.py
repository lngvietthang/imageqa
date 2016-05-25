import os
import argparse

import numpy as np
import pickle

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform, choice

from itertools import izip_longest


verbose = os.environ.get('VERBOSE', 'no') == 'yes'
debug = os.environ.get('DEBUG', 'no') == 'yes'


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def load_data():
    data_train = np.load(os.path.join(path2indir, 'train.npy'))
    q_train = data_train[0][:, 1:]
    a_train = data_train[1]

    data_dev = np.load(os.path.join(path2indir, 'dev.npy'))
    q_dev = data_dev[0][:, 1:]
    a_dev = data_dev[1]

    data_val = np.load(os.path.join(path2indir, 'val.npy'))
    q_val = data_val[0][:, 1:]
    a_val = data_val[1]

    fread = open(os.path.join(path2indir, 'qdict.pkl'))
    qdict = pickle.load(fread)
    fread.close()
    fread = open(os.path.join(path2indir, 'adict.pkl'))
    adict = pickle.load(fread)
    fread.close()

    return q_train, q_dev, q_val, a_train, a_dev, a_val, qdict, adict


def model(q_train, q_dev, q_val, a_train, a_dev, a_val, qdict, adict):
    vocab_size = len(qdict)
    nb_ans = len(adict)

    nb_epoch = 1000

    quest_model = Sequential()
    quest_model.add(Embedding(input_dim=vocab_size, output_dim={{choice([100, 200, 300, 500])}},
                              init = {{choice(['uniform', 'normal', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform'])}},
                              mask_zero=True, dropout={{uniform(0,1)}}
                              )
                    )
    quest_model.add(Lambda(function=lambda x: K.sum(x, axis=1), output_shape=lambda shape: (shape[0], ) + shape[2:]))
    quest_model.add(Dense(nb_ans))
    quest_model.add(Activation('softmax'))

    quest_model.compile(loss='categorical_crossentropy',
                        optimizer={{choice(['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax'])}},
                        metrics=['accuracy'])

    print 'Train...'
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5', verbose=1, save_best_only=True)
    quest_model.fit(q_train, a_train, batch_size={{choice([32, 64, 100])}}, nb_epoch=nb_epoch,
                    validation_data=(q_dev, a_dev),
                    callbacks=[early_stopping, checkpointer])

    score, acc = quest_model.evaluate(q_val, a_val, verbose=1)

    print 'Test accuracy:', acc

    return {'loss': -acc, 'status': STATUS_OK, 'model': quest_model}
    # best_model = None
    # best_acc = None
    # wait = 0
    # stop = False
    # for ie in xrange(nb_epoch):
    #     print 'Epoch %3d:' % ie
    #     progbar = generic_utils.Progbar(len(q_train))
    #
    #     for qb, ab in zip(grouper(q_train, batch_size), grouper(a_train, batch_size)):
    #         if qb[-1] is None:
    #             qb = [q for q in qb if q is not None]
    #         if ab[-1] is None:
    #             ab = [a for a in ab if a is not None]
    #
    #         ab = np_utils.to_categorical(ab, nb_ans)
    #
    #         train_loss = quest_model.train_on_batch(qb, ab)
    #
    #         progbar.add(len(qb), values=[('train loss', train_loss[0])])
    #
    #     print 'Validation...'
    #     av = np_utils.to_categorical(a_dev, nb_ans)
    #
    #     val_loss, val_acc = quest_model.evaluate(q_dev, av, batch_size=batch_size, show_accuracy=True)
    #     print "--> Validation Accuracy: %.4f" % (val_acc)
    #     if (best_acc is None) or (val_acc > best_acc):
    #         best_acc = val_acc
    #         wait = 0
    #         best_model = quest_model
    #     else:
    #         wait += 1
    #         if wait > patience:
    #             print 'Patience level reached, early stop.'
    #             stop = True
    #     if stop:
    #         break



def main():
    # TODO: Parsing the list arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir', required=True, type=str)
    parser.add_argument('-outdir', required=True, type=str)
    args = parser.parse_args()

    global path2indir
    path2indir = args.indir
    global path2outputdir
    path2outputdir = args.outdir

    best_run, best_model = optim.minimize(model=model,
                                          data=load_data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    print best_run

if __name__ == '__main__':
    main()