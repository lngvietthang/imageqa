from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform, choice

import argparse


def data():
    import os
    import numpy as np
    import pickle
    from keras.utils import np_utils

    path2indir = '/home/guest/Development/myprojects/cocoqa-dataset/prepared/'

    data_train = np.load(os.path.join(path2indir, 'train.npy'))
    q_train = data_train[0][:, 1:]
    a_train = data_train[1][:]

    data_dev = np.load(os.path.join(path2indir, 'dev.npy'))
    q_dev = data_dev[0][:, 1:]
    a_dev = data_dev[1][:]

    data_val = np.load(os.path.join(path2indir, 'val.npy'))
    q_val = data_val[0][:, 1:]
    a_val = data_val[1][:]

    fread = open(os.path.join(path2indir, 'qdict.pkl'))
    qdict = pickle.load(fread)
    fread.close()
    fread = open(os.path.join(path2indir, 'adict.pkl'))
    adict = pickle.load(fread)
    fread.close()

    nb_ans = len(adict)
    a_train = np_utils.to_categorical(a_train, nb_ans)
    a_dev = np_utils.to_categorical(a_dev, nb_ans)
    a_val = np_utils.to_categorical(a_val, nb_ans)

    return q_train, q_dev, q_val, a_train, a_dev, a_val, qdict, adict


def model(q_train, q_dev, q_val, a_train, a_dev, a_val, qdict, adict):
    from keras.models import Sequential
    from keras.layers.embeddings import Embedding
    from keras.layers.core import Dense, Activation
    from keras.layers.recurrent import LSTM
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    vocab_size = len(qdict)
    nb_ans = len(adict)

    nb_epoch = 1000

    quest_model = Sequential()
    quest_model.add(Embedding(input_dim=vocab_size, output_dim={{choice([100, 200, 300, 500])}},
                              init = {{choice(['uniform', 'lecun_uniform', 'normal',
                                               'identity', 'glorot_uniform', 'glorot_normal',
                                               'he_normal', 'he_uniform'])}},
                              mask_zero=True, dropout={{uniform(0, 1)}}
                              )
                    )
#    nb_ltsmlayer = {{choice([1, 2, 3, 4])}}
    nb_ltsmlayer = 1

    if nb_ltsmlayer == 1:
        quest_model.add(LSTM(output_dim={{choice([100, 200, 300, 500])}},
                             init={{choice(['uniform', 'lecun_uniform', 'normal',
                                            'identity', 'glorot_uniform', 'glorot_normal',
                                            'orthogonal', 'he_normal', 'he_uniform'])}},
                             inner_init={{choice(['uniform', 'lecun_uniform', 'normal',
                                                  'identity', 'glorot_uniform', 'glorot_normal',
                                                  'orthogonal', 'he_normal', 'he_uniform'])}},
                             activation={{choice(['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])}},
                             inner_activation={{choice(['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])}},
                             W_regularizer={{choice(['l1', 'l2', 'l1l2'])}},
                             U_regularizer={{choice(['l1', 'l2', 'l1l2'])}},
                             b_regularizer={{choice(['l1', 'l2', 'l1l2'])}},
                             dropout_W={{uniform(0, 1)}},
                             dropout_U={{uniform(0, 1)}},
                             return_sequences=False))
    else:
        for i in range(nb_ltsmlayer-1):
            quest_model.add(LSTM(output_dim={{choice([100, 200, 300, 500])}},
                                 init={{choice(['uniform', 'lecun_uniform', 'normal',
                                                'identity', 'glorot_uniform', 'glorot_normal',
                                                'orthogonal', 'he_normal', 'he_uniform'])}},
                                 inner_init={{choice(['uniform', 'lecun_uniform', 'normal',
                                                      'identity', 'glorot_uniform', 'glorot_normal',
                                                      'orthogonal', 'he_normal', 'he_uniform'])}},
                                 activation={{choice(['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])}},
                                 inner_activation={{choice(['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])}},
                                 W_regularizer={{choice(['l1', 'l2', 'l1l2'])}},
                                 U_regularizer={{choice(['l1', 'l2', 'l1l2'])}},
                                 b_regularizer={{choice(['l1', 'l2', 'l1l2'])}},
                                 dropout_W={{uniform(0, 1)}},
                                 dropout_U={{uniform(0, 1)}},
                                 return_sequences=True))

        quest_model.add(LSTM(output_dim={{choice([100, 200, 300, 500])}},
                             init={{choice(['uniform', 'lecun_uniform', 'normal',
                                            'identity', 'glorot_uniform', 'glorot_normal',
                                            'orthogonal', 'he_normal', 'he_uniform'])}},
                             inner_init={{choice(['uniform', 'lecun_uniform', 'normal',
                                                  'identity', 'glorot_uniform', 'glorot_normal',
                                                  'orthogonal', 'he_normal', 'he_uniform'])}},
                             activation={{choice(['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])}},
                             inner_activation={{choice(['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])}},
                             W_regularizer={{choice(['l1', 'l2', 'l1l2'])}},
                             U_regularizer={{choice(['l1', 'l2', 'l1l2'])}},
                             b_regularizer={{choice(['l1', 'l2', 'l1l2'])}},
                             dropout_W={{uniform(0, 1)}},
                             dropout_U={{uniform(0, 1)}},
                             return_sequences=False))

    quest_model.add(Dense(nb_ans))
    quest_model.add(Activation('softmax'))

    quest_model.compile(loss='categorical_crossentropy',
                        optimizer={{choice(['adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax'])}},
                        metrics=['accuracy'])

    print('##################################')
    print('Train...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5', verbose=1, save_best_only=True)
    quest_model.fit(q_train, a_train, batch_size={{choice([32, 64, 100])}}, nb_epoch=nb_epoch,
                    validation_data=(q_dev, a_dev),
                    callbacks=[early_stopping, checkpointer])

    score, acc = quest_model.evaluate(q_val, a_val, verbose=1)

    print('##################################')
    print('Test accuracy:%.4f' % acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': quest_model}


def main():
    # TODO: Parsing the list arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir', required=True, type=str)
    parser.add_argument('-outdir', required=True, type=str)
    args = parser.parse_args()

    global path2indir
    path2indir = args.indir
#    global path2outputdir
#    path2outputdir = args.outdir

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    print(best_run)

if __name__ == '__main__':
    main()