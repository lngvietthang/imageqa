from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform, choice

import argparse


def data():
    import os
    import numpy as np
    import pickle
    from scipy import sparse
    import h5py
    from keras.utils import np_utils

    path2indir = '/home/guest/Development/myprojects/cocoqa-dataset/prepared/'
    h5key = 'vgg-fc7'

    data_train = np.load(os.path.join(path2indir, 'train.npy'))
    q_train = data_train[0][:, 1:]
    i_train = data_train[0][:, 0]
    a_train = data_train[1][:]

    data_dev = np.load(os.path.join(path2indir, 'dev.npy'))
    q_dev = data_dev[0][:, 1:]
    i_dev = data_dev[0][:, 0]
    a_dev = data_dev[1][:]

    data_val = np.load(os.path.join(path2indir, 'val.npy'))
    q_val = data_val[0][:, 1:]
    i_val = data_val[0][:, 0]
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

    img_feat_sparse = h5py.File(os.path.join(path2indir, 'image-features-all.h5'))
    img_feat_shape = img_feat_sparse[h5key + '_shape'][:]
    img_feat_data = img_feat_sparse[h5key + '_data']
    img_feat_indices = img_feat_sparse[h5key + '_indices']
    img_feat_indptr = img_feat_sparse[h5key + '_indptr']

    img_feat = sparse.csr_matrix((img_feat_data, img_feat_indices, img_feat_indptr), shape=img_feat_shape)
    img_feat = img_feat.toarray()

    img_mean = img_feat_sparse[h5key + '_mean']
    img_std = img_feat_sparse[h5key + '_std']

    img_feat = (img_feat - img_mean) / img_std
    del img_feat_sparse, img_mean, img_std

    img_feat_train = np.zeros((len(i_train), img_feat.shape[1]), dtype='float32')
    for idx, img_id in enumerate(i_train): img_feat_train[idx] = img_feat[img_id][:]

    img_feat_dev = np.zeros((len(i_dev), img_feat.shape[1]), dtype='float32')
    for idx, img_id in enumerate(i_dev): img_feat_dev[idx] = img_feat[img_id][:]

    img_feat_val = np.zeros((len(i_val), img_feat.shape[1]), dtype='float32')
    for idx, img_id in enumerate(i_val): img_feat_val[idx] = img_feat[img_id][:]

    del img_feat

    return img_feat_train, img_feat_dev, img_feat_val, q_train, q_dev, q_val, a_train, a_dev, a_val, qdict, adict


def model(img_feat_train, img_feat_dev, img_feat_val, q_train, q_dev, q_val, a_train, a_dev, a_val, qdict, adict):
    from keras.models import Sequential
    from keras.layers.embeddings import Embedding
    from keras.layers.core import Lambda, Dense, Activation, Merge, Dropout, Reshape
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import keras.backend as K

    vocab_size = len(qdict)
    nb_ans = len(adict)

    nb_epoch = 1000

    quest_model = Sequential()
    quest_model.add(Embedding(input_dim=vocab_size, output_dim={{choice([100, 200, 300, 500])}},
                              init={{choice(['uniform', 'normal', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform'])}},
                              mask_zero=False, dropout={{uniform(0,1)}}
                              )
                    )
    quest_model.add(Lambda(function=lambda x: K.sum(x, axis=1), output_shape=lambda shape: (shape[0], ) + shape[2:]))

    nb_feat = img_feat_train.shape[1]
    img_model = Sequential()
    img_model.add(Reshape((nb_feat, ), input_shape=(nb_feat,)))
    img_model.add(Dropout({{uniform(0, 1)}}))

    multimodal = Sequential()
    multimodal.add(Merge([img_model, quest_model], mode='concat', concat_axis=1))
    multimodal.add(Dense(nb_ans))
    multimodal.add(Activation('softmax'))

    multimodal.compile(loss='categorical_crossentropy',
                       optimizer={{choice(['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax'])}},
                       metrics=['accuracy'])

    print('##################################')
    print('Train...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath='cnn_bow_weights.hdf5', verbose=1, save_best_only=True)
    multimodal.fit([img_feat_train, q_train], a_train, batch_size={{choice([32])}}, nb_epoch=nb_epoch,
                   validation_data=([img_feat_dev, q_dev], a_dev),
                   callbacks=[early_stopping, checkpointer])

    score, acc = multimodal.evaluate([img_feat_val, q_val], a_val, verbose=1)

    print('##################################')
    print('Test accuracy:%.4f' % acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': multimodal}


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