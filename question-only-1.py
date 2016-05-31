from __future__ import print_function

import argparse
import os
import numpy as np
import pickle
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K


def data(path2indir):
    data_dev1 = np.load(os.path.join(path2indir, 'dev1.npy'))
    q_dev1 = data_dev1[0][:, 1:]
    a_dev1 = data_dev1[1][:]

    data_dev2 = np.load(os.path.join(path2indir, 'dev2.npy'))
    q_dev2 = data_dev2[0][:, 1:]
    a_dev2 = data_dev2[1][:]

    data_val = np.load(os.path.join(path2indir, 'val.npy'))
    q_val = data_val[0][:, 1:]
    a_val = data_val[1][:]

    with open(os.path.join(path2indir, 'qdict-dev1.pkl')) as fread:
        qdict_dev1 = pickle.load(fread)

    with open(os.path.join(path2indir, 'adict-dev1.pkl')) as fread:
        adict_dev1 = pickle.load(fread)

    nb_ans = len(adict_dev1) - 1
    a_dev1 = np_utils.to_categorical(a_dev1, nb_ans)
    a_dev2 = np_utils.to_categorical(a_dev2, nb_ans)
    a_val = np_utils.to_categorical(a_val, nb_ans)

    return q_dev1, q_dev2, q_val, a_dev1, a_dev2, a_val, qdict_dev1, adict_dev1


def model(q_dev1, q_dev2, q_val, a_dev1, a_dev2, a_val, qdict_dev1, adict_dev1, path2outdir):
    vocab_size = len(qdict_dev1)
    nb_ans = len(adict_dev1) - 1

    nb_epoch = 1000

    quest_model = Sequential()
    quest_model.add(Embedding(input_dim=vocab_size, output_dim= 200,
                              init = 'glorot_uniform',
                              mask_zero=False, dropout=0.2
                              )
                    )
    quest_model.add(Lambda(function=lambda x: K.sum(x, axis=1), output_shape=lambda shape: (shape[0], ) + shape[2:]))
    quest_model.add(Dense(nb_ans))
    quest_model.add(Activation('softmax'))

    quest_model.compile(loss='categorical_crossentropy',
                        optimizer='adamax',
                        metrics=['accuracy'])

    print('##################################')
    print('Train...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath=os.path.join(path2outdir, 'keras_weights.hdf5'), verbose=1, save_best_only=True)
    quest_model.fit(q_dev1, a_dev1, batch_size=100, nb_epoch=nb_epoch,
                    validation_data=(q_dev2, a_dev2),
                    callbacks=[early_stopping, checkpointer])

    print('##################################')
    print('Testing')
    quest_model.load_weights(filepath=os.path.join(path2outdir, 'keras_weights.hdf5'))
    score, acc = quest_model.evaluate(q_val, a_val, verbose=1)
    result = quest_model.predict(q_val, verbose=1)

    np.save(os.path.join(path2outdir, 'bow-results.npy'), result, dtype=object)

    print('Test accuracy:', acc)


def main():
    # TODO: Parsing the list arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir', required=True, type=str)
    parser.add_argument('-outdir', required=True, type=str)
    args = parser.parse_args()

    path2indir = args.indir
    path2outdir = args.outdir

    q_dev1, q_dev2, q_val, a_dev1, a_dev2, a_val, qdict_dev1, adict_dev1 = data(path2indir)

    model(q_dev1, q_dev2, q_val, a_dev1, a_dev2, a_val, qdict_dev1, adict_dev1, path2outdir)

if __name__ == '__main__':
    main()