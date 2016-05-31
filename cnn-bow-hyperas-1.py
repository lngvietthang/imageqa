from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform, choice


def data():
    import os
    import numpy as np
    import pickle
    from scipy import sparse
    import h5py
    from keras.utils import np_utils

    path2indir = os.environ.get('INDIR', 'no')
    img_h5key = os.environ.get('H5KEY', 'no')
    sparse = os.environ.get('SPARSE', 'yes') == 'yes'

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

    fread = open(os.path.join(path2indir, 'qdict-dev1.pkl'))
    qdict_dev1 = pickle.load(fread)
    fread.close()
    fread = open(os.path.join(path2indir, 'adict-dev1.pkl'))
    adict_dev1 = pickle.load(fread)
    fread.close()

    nb_ans = len(adict_dev1) - 1
    a_dev1 = np_utils.to_categorical(a_dev1, nb_ans)
    a_dev2 = np_utils.to_categorical(a_dev2, nb_ans)
    a_val = np_utils.to_categorical(a_val, nb_ans)

    if sparse:
        img_feat_sparse = h5py.File(os.path.join(path2indir, img_h5key + '.h5'))
        img_feat_shape = img_feat_sparse[img_h5key + '_shape'][:]
        img_feat_data = img_feat_sparse[img_h5key + '_data']
        img_feat_indices = img_feat_sparse[img_h5key + '_indices']
        img_feat_indptr = img_feat_sparse[img_h5key + '_indptr']

        img_feat = sparse.csr_matrix((img_feat_data, img_feat_indices, img_feat_indptr), shape=img_feat_shape)
        img_feat = img_feat.toarray()

        img_mean = img_feat_sparse[img_h5key + '_mean']
        img_std = img_feat_sparse[img_h5key + '_std']

        img_feat = (img_feat - img_mean) / img_std
        del img_feat_sparse, img_mean, img_std
    else:
        img_feat_h5 = h5py.File(os.path.join(path2indir, img_h5key + '.h5'))
        img_feat = img_feat_h5[img_h5key + '_data']
        img_mean = img_feat_h5[img_h5key + '_mean']
        img_std = img_feat_h5[img_h5key + '_std']

        img_feat = (img_feat - img_mean) / img_std
        del img_feat_h5, img_mean, img_std

    img_feat_dev1 = np.zeros((len(i_dev1), img_feat.shape[1]), dtype='float32')
    for idx, img_id in enumerate(i_dev1): img_feat_dev1[idx] = img_feat[img_id][:]

    img_feat_dev2 = np.zeros((len(i_dev2), img_feat.shape[1]), dtype='float32')
    for idx, img_id in enumerate(i_dev2): img_feat_dev2[idx] = img_feat[img_id][:]

    img_feat_val = np.zeros((len(i_val), img_feat.shape[1]), dtype='float32')
    for idx, img_id in enumerate(i_val): img_feat_val[idx] = img_feat[img_id][:]

    del img_feat

    return img_feat_dev1, img_feat_dev2, img_feat_val, q_dev1, q_dev2, q_val, a_dev1, a_dev2, a_val, qdict_dev1, adict_dev1


def model(img_feat_dev1, img_feat_dev2, img_feat_val, q_dev1, q_dev2, q_val, a_dev1, a_dev2, a_val, qdict_dev1, adict_dev1):
    from keras.models import Sequential
    from keras.layers.embeddings import Embedding
    from keras.layers.core import Lambda, Dense, Activation, Merge, Dropout, Reshape
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import keras.backend as K
    import os

    path2outdir = os.environ.get('OUTDIR', 'no')

    vocab_size = len(qdict_dev1)
    nb_ans = len(adict_dev1) - 1

    nb_epoch = 1000

    quest_model = Sequential()
    quest_model.add(Embedding(input_dim=vocab_size, output_dim={{choice([100, 200, 300, 500])}},
                              init={{choice(['uniform', 'normal', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform'])}},
                              mask_zero=False, dropout={{uniform(0,1)}}
                              )
                    )
    quest_model.add(Lambda(function=lambda x: K.sum(x, axis=1), output_shape=lambda shape: (shape[0], ) + shape[2:]))

    nb_feat = img_feat_dev1.shape[1]
    img_model = Sequential()
    img_model.add(Reshape((nb_feat, ), input_shape=(nb_feat,)))

    multimodal = Sequential()
    multimodal.add(Merge([img_model, quest_model], mode='concat', concat_axis=1))
    multimodal.add(Dropout({{uniform(0, 1)}}))
    multimodal.add(Dense(nb_ans))
    multimodal.add(Activation('softmax'))

    multimodal.compile(loss='categorical_crossentropy',
                       optimizer={{choice(['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax'])}},
                       metrics=['accuracy'])

    print('##################################')
    print('Train...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath=os.path.join(path2outdir, 'cnn_bow_weights.hdf5'), verbose=1, save_best_only=True)
    multimodal.fit([img_feat_dev1, q_dev1], a_dev1, batch_size={{choice([32, 64, 100])}}, nb_epoch=nb_epoch,
                   validation_data=([img_feat_dev2, q_dev2], a_dev2),
                   callbacks=[early_stopping, checkpointer])
    multimodal.load_weights(os.path.join(path2outdir, 'cnn_bow_weights.hdf5'))
    score, acc = multimodal.evaluate([img_feat_val, q_val], a_val, verbose=1)

    print('##################################')
    print('Test accuracy:%.4f' % acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': multimodal}


def main():
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())

    print(best_run)

if __name__ == '__main__':
    main()