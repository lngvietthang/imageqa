from __future__ import print_function
import argparse
import os
import numpy as np
import pickle
from scipy import sparse
import h5py
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Activation, Merge, Dropout, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K


def data(path2indir, h5key, sparse):
    data_dev1 = np.load(os.path.join(path2indir, 'dev1.npy'))
    q_dev1 = data_dev1[0][:, 1:]
    i_dev1 = data_dev1[0][:, 0]
    a_dev1 = data_dev1[1][:]

    data_dev2 = np.load(os.path.join(path2indir, 'dev2.npy'))
    q_dev2 = data_dev2[0][:, 1:]
    i_dev2 = data_dev2[0][:, 0]
    a_dev2 = data_dev2[1][:]

    data_val = np.load(os.path.join(path2indir, 'val.npy'))
    q_val = data_val[0][:, 1:]
    i_val = data_val[0][:, 0]
    a_val = data_val[1][:]

    with open(os.path.join(path2indir, 'qdict-dev1.pkl')) as fread:
        qdict_dev1 = pickle.load(fread)

    with open(os.path.join(path2indir, 'adict-dev1.pkl')) as fread:
        adict_dev1 = pickle.load(fread)

    nb_ans = len(adict_dev1) - 1
    a_dev1 = np_utils.to_categorical(a_dev1, nb_ans)
    a_dev2 = np_utils.to_categorical(a_dev2, nb_ans)
    a_val = np_utils.to_categorical(a_val, nb_ans)

    if sparse:
        depnet_feat_sparse = h5py.File(os.path.join(path2indir, h5key + '.h5'))
        depnet_feat_shape = depnet_feat_sparse[h5key + '_shape'][:]
        depnet_feat_data = depnet_feat_sparse[h5key + '_data']
        depnet_feat_indices = depnet_feat_sparse[h5key + '_indices']
        depnet_feat_indptr = depnet_feat_sparse[h5key + '_indptr']

        depnet_feat = sparse.csr_matrix((depnet_feat_data, depnet_feat_indices, depnet_feat_indptr), shape=depnet_feat_shape)
        depnet_feat = depnet_feat.toarray()

        depnet_mean = depnet_feat_sparse[h5key + '_mean']
        depnet_std = depnet_feat_sparse[h5key + '_std']

        depnet_feat = (depnet_feat - depnet_mean) / depnet_std
        del depnet_feat_sparse, depnet_mean, depnet_std
    else:
        depnet_feat_h5 = h5py.File(os.path.join(path2indir, h5key + '.h5'))
        depnet_feat = depnet_feat_h5[h5key + '_data'][:]
        depnet_mean = depnet_feat_h5[h5key + '_mean'][:]
        depnet_std = depnet_feat_h5[h5key + '_std'][:]

        depnet_feat = (depnet_feat - depnet_mean) / depnet_std
        del depnet_feat_h5, depnet_mean, depnet_std

    depnet_feat_dev1 = np.zeros((len(i_dev1), depnet_feat.shape[1]), dtype='float32')
    for idx, img_id in enumerate(i_dev1):
        depnet_feat_dev1[idx] = depnet_feat[img_id][:]

    depnet_feat_dev2 = np.zeros((len(i_dev2), depnet_feat.shape[1]), dtype='float32')
    for idx, img_id in enumerate(i_dev2):
        depnet_feat_dev2[idx] = depnet_feat[img_id][:]

    depnet_feat_val = np.zeros((len(i_val), depnet_feat.shape[1]), dtype='float32')
    for idx, img_id in enumerate(i_val):
        depnet_feat_val[idx] = depnet_feat[img_id][:]

    del depnet_feat

    return depnet_feat_dev1, depnet_feat_dev2, depnet_feat_val, q_dev1, q_dev2, q_val, a_dev1, a_dev2, a_val, qdict_dev1, adict_dev1


def model(depnet_feat_dev1, depnet_feat_dev2, depnet_feat_val, q_dev1, q_dev2, q_val, a_dev1, a_dev2, a_val, qdict_dev1, adict_dev1, path2outdir, h5key):
    vocab_size = len(qdict_dev1)
    nb_ans = len(adict_dev1) - 1

    nb_epoch = 1000

    quest_model = Sequential()
    quest_model.add(Embedding(input_dim=vocab_size, output_dim=500,
                              init='glorot_normal',
                              mask_zero=False, dropout=0.15
                              )
                    )
    quest_model.add(Lambda(function=lambda x: K.sum(x, axis=1), output_shape=lambda shape: (shape[0], ) + shape[2:]))

    nb_feat = depnet_feat_dev1.shape[1]
    depnet_model = Sequential()
    depnet_model.add(Reshape((nb_feat, ), input_shape=(nb_feat,)))

    multimodal = Sequential()
    multimodal.add(Merge([depnet_model, quest_model], mode='concat', concat_axis=1))
    multimodal.add(Dropout(0.86))
    multimodal.add(Dense(nb_ans))
    multimodal.add(Activation('softmax'))

    multimodal.compile(loss='categorical_crossentropy',
                       optimizer='adagrad',
                       metrics=['accuracy'])

    print('##################################')
    print('Train...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath=os.path.join(path2outdir, 'cnn_bow_weights.hdf5'), verbose=1, save_best_only=True)
    multimodal.fit([depnet_feat_dev1, q_dev1], a_dev1, batch_size=64, nb_epoch=nb_epoch,
                   validation_data=([depnet_feat_dev2, q_dev2], a_dev2),
                   callbacks=[early_stopping, checkpointer])
    print('##################################')
    print('Test...')
    multimodal.load_weights(os.path.join(path2outdir, 'cnn_bow_weights.hdf5'))
    score, acc = multimodal.evaluate([depnet_feat_val, q_val], a_val, verbose=1)
    result = multimodal.predict([depnet_feat_val, q_val], verbose=1)
    np.save(os.path.join(path2outdir, h5key + '-bow-results.npy'), result)
    print('##################################')
    print('Test accuracy:%.4f' % acc)


def main():
    # TODO: Parsing the list arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir', required=True, type=str)
    parser.add_argument('-outdir', required=True, type=str)
    parser.add_argument('-h5key', required=True, type=str)
    parser.add_argument('-sparse', default=False, type=bool)
    args = parser.parse_args()

    path2indir = args.indir
    path2outdir = args.outdir
    h5key = args.h5key
    sparse = args.sparse

    img_feat_train, img_feat_dev, img_feat_val, q_train, q_dev, q_val, a_train, a_dev, a_val, qdict, adict = data(path2indir, h5key, sparse)

    model(img_feat_train, img_feat_dev, img_feat_val, q_train, q_dev, q_val, a_train, a_dev, a_val, qdict, adict, path2outdir, h5key)


if __name__ == '__main__':
    main()