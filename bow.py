# coding: utf-8
"""Train Question Bag-of-word classifier"""

from __future__ import print_function
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../utils')

import argparse
import numpy
import logging
from datetime import datetime
import h5py
import json
import tempfile

import vocabulary
import csrmatrix

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Activation
import keras.backend as K
from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

logger = logging.getLogger(__name__)

model_filename = 'bow.json'
param_filename = 'bow_param.h5'
vocab_filename = 'ans_vocab.json'


######################################################################

# def mlp(input_size, output_size, num_units=1024, activation='relu', dropout=0.5, l2=0.0, optimizer='adam'):
#     logger.info(
#         'num_units={}, activation={}, dropout={}, l2={}, optimizer={}'.format(num_units, activation, dropout, l2,
#                                                                               optimizer))
#     reg_l2 = regularizers.l2(l2) if l2 > 0.0 else None
#     model = Sequential()
#     model.add(Dense(num_units, input_dim=input_size, activation=activation, W_regularizer=reg_l2))
#     # model.add(BatchNormalization(input_shape=(input_size,), mode=0))
#     # model.add(Dense(num_units, activation=activation, W_regularizer=reg_l2))
#     if dropout > 0.0: model.add(Dropout(dropout))
#     # model.add(Dense(512, activation='relu'))
#     # model.add(Dropout(0.5))
#     model.add(Dense(output_size, activation='softmax', W_regularizer=reg_l2))
#     model.compile(loss='categorical_crossentropy',
#                   metrics=['accuracy'],
#                   optimizer=optimizer)
#     return model
#
#
# def loglinear(input_size, output_size, l2=0.0, optimizer='adam'):
#     logger.info('l2={}, optimizer={}'.format(l2, optimizer))
#     reg_l2 = regularizers.l2(l2) if l2 > 0.0 else None
#     model = Sequential()
#     model.add(Dense(output_size, input_dim=input_size, activation='softmax', W_regularizer=reg_l2))
#     model.compile(loss='categorical_crossentropy',
#                   metrics=['accuracy'],
#                   optimizer=optimizer)
#     return model


# callback to flush stdout - in order to see log messages timely
class FlushStdout(Callback):
    def on_batch_begin(self, batch, logs={}):
        sys.stdout.flush()

    def on_train_end(self, logs={}):
        sys.stdout.flush()


def train_bow(x, y, validation_data, vocab_size, nb_ans, num_iter, batch_size=1, on_memory=True, early_stopping=-1,
              wemb_dim=200, wemb_init='glorot_uniform', wemb_dropout=0.2, optimizer='adamax'):
    # TODO: Support Masking
    # -> Use the class in gist (https://gist.github.com/lngvietthang/67f35c1ee4481284dfcd9b34c7fe1fc6)
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=wemb_dim,
                        init=wemb_init, mask_zero=False, dropout=wemb_dropout))
    model.add(Lambda(function=lambda x: K.sum(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:]))
    model.add(Dense(nb_ans))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=early_stopping)] if early_stopping >= 0 else []
    temp_model = tempfile.NamedTemporaryFile()
    callbacks.append(ModelCheckpoint(filepath=temp_model.name, save_best_only=True))
    callbacks.append(FlushStdout())
    shuffle = True if on_memory else 'batch'
    model.fit(x, y, batch_size=batch_size, nb_epoch=num_iter, validation_data=validation_data, shuffle=shuffle,
              callbacks=callbacks, verbose=2)
    sys.stdout.flush()

    model.load_weights(temp_model.name)
    loss, acc = model.evaluate(validation_data[0], validation_data[1], batch_size=batch_size)
    temp_model.close()

    return model, loss, acc


# def train_keras(x, y, model_name, validation_data, num_iter, batch_size=1, on_memory=True, early_stopping=-1,
#                 num_jobs=1, **model_params):
#     num_labels = y.shape[1]
#     feature_size = x.shape[1]
#     if model_name == 'loglinear':
#         model = loglinear(feature_size, num_labels, **model_params)
#     else:
#         assert (model_name == 'mlp')
#         model = mlp(feature_size, num_labels, **model_params)
#     shuffle = True if on_memory else 'batch'
#     callbacks = [EarlyStopping(monitor='val_loss', patience=early_stopping)] if early_stopping >= 0 else []
#     temp_model = tempfile.NamedTemporaryFile()
#     callbacks.append(ModelCheckpoint(filepath=temp_model.name, save_best_only=True))
#     callbacks.append(FlushStdout())
#     history = model.fit(x, y, validation_data=validation_data, nb_epoch=num_iter, batch_size=batch_size, verbose=2,
#                         shuffle=shuffle, callbacks=callbacks)
#     # print(history.history)
#     sys.stdout.flush()
#     model.load_weights(temp_model.name)
#     if validation_data is not None:
#         loss, acc = model.evaluate(validation_data[0], validation_data[1], batch_size=batch_size)
#     else:
#         loss, acc = 0.0, 0.0
#     temp_model.close()
#     return model, loss, acc


# def onehot2label(mat):
#     """convert one-hot representation into integer labels"""
#     labels = [numpy.argmax(l) for l in mat]
#     return labels
#
#
# def train_sklearn(x, y, model_name, validation_data, num_iter, on_memory=True,
#                   num_jobs=1, l2=0.0001):
#     assert (model_name == 'loglinear')
#     assert (l2 > 0.0)
#     logger.info('l2={}'.format(l2))
#     assert (len(y.shape) == 2)
#     model = OneVsRestClassifier(
#         SGDClassifier(loss='log', penalty='l2', alpha=l2, n_iter=num_iter, verbose=2))  # , n_jobs=num_jobs)
#     # model = LogisticRegression(penalty='l2', C=l2, max_iter=num_iter, multi_class='multinomial', solver='lbfgs', verbose=2)
#     model.fit(x, y)
#     pred_y = model.predict_proba(validation_data[0])
#     val_loss = log_loss(validation_data[1], pred_y)
#     val_acc = accuracy_score(onehot2label(validation_data[1]), onehot2label(pred_y))
#     return model, val_loss, val_acc  # TODO: return val_loss and val_acc


# def generate_params(grid):
#     if grid is None: return [{}]
#     assert (isinstance(grid, dict))
#     keys = grid.keys()
#     values = grid.values()
#     return [dict(zip(keys, params)) for params in itertools.product(*values)]

def train(x, y, validation_data, vocab_size, nb_ans, num_iter, batch_size=1, on_memory=True, early_stopping=-1,
          hyperopt_params=None, **model_params):
    best_model = None
    best_loss = None
    best_acc = None

    logger.info('Start training: {} hyperparameters searching'.format(
        len(hyperopt_params) if hyperopt_params is not None else 0))
    if hyperopt_params is None:
        best_model, best_loss, best_acc = train_bow(x, y, validation_data, vocab_size, nb_ans, num_iter, batch_size,
                                                    on_memory, early_stopping, **model_params)
        best_params = model_params
        logs = [best_params.values() + [best_loss, best_acc]]
    else:
        def keras_fmin_fnct(space):
            """
            :param space:
            :return:
            """

            if not hasattr(keras_fmin_fnct, "counter"):
                keras_fmin_fnct.counter = 0
            keras_fmin_fnct.counter += 1

            model, loss, acc = train_bow(x, y, validation_data, vocab_size, nb_ans, num_iter, batch_size, on_memory,
                                         early_stopping, **space)

            logger.info('Searching hyperparameter {} done: loss={}, acc={}'.format(keras_fmin_fnct.counter, loss, acc))

            return {'loss': -acc, 'status': STATUS_OK, 'model': model, 'model_loss': loss, 'model_acc': acc}

        def get_space():
            return {
                'wemb_dim': hp.choice('wemb_dim', hyperopt_params['wembdim']),
                'wemb_init': hp.choice('wemb_init', hyperopt_params['wembinit']),
                'wemb_dropout': hp.choice('wemb_dropout', hyperopt_params['wembdropout']),
                'optimizer': hp.choice('optimizer', hyperopt_params['optimizer']),
            }

        trials = Trials()
        best_params = fmin(keras_fmin_fnct, space=get_space(), algo=tpe.suggest,
                           max_evals=hyperopt_params['maxevals'], trials=trials)
        logs = []
        for trial in trials:
            vals = trial.get('misc').get('vals')
            for key in vals.keys():
                vals[key] = vals[key][0]
            logs.append(vals.values() + [trial.get('result').get('model_loss'),
                                         trial.get('result').get('model_acc')])
            if trial.get('misc').get('vals') == best_params and 'model' in trial.get('result').keys():
                best_model = trial.get('result').get('model')
                best_loss = trial.get('result').get('model_loss')
                best_acc = trial.get('result').get('model_acc')

    logger.info('Best loss={}, acc={}, model: {}'.format(best_loss, best_acc, best_params))
    logger.info('{}\tloss\tacc'.format('\t'.join(best_params.keys())))
    for log in logs:
        logger.info('\t'.join([str(l) for l in log]))

    return best_model, best_loss, best_acc


######################################################################

def check_train_data(x, y):
    if len(x.shape) != 2:
        raise ValueError("Training data x must be 2d array")
    if len(y.shape) != 2:
        raise ValueError("Target labels y must be 2d array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("1st axis of x and y must have the same size")
    return True


######################################################################

def precision_recall_fscore(correct, pred, gold):
    if pred == 0:
        precision = 0.0
    else:
        precision = float(correct) / pred
    if gold == 0:
        recall = 0.0
    else:
        recall = float(correct) / gold
    if precision + recall == 0.0:
        fscore = 0.0
    else:
        fscore = (2 * precision * recall) / (precision + recall)
    return precision, recall, fscore


def show_accuracy(pred_y, val_y, val_index, vocab):
    correct = {}
    total_pred = {}
    total_gold = {}
    for (pred, gold, sample_id) in zip(pred_y, val_y, val_index):
        pred_label = numpy.argmax(pred)
        gold_label = numpy.argmax(gold)
        logger.debug('%s:', sample_id)
        logger.debug('  pred: %s', vocab.invget(pred_label))
        logger.debug('  gold: %s', vocab.invget(gold_label))
        total_pred[pred_label] = total_pred.get(pred_label, 0) + 1
        total_gold[gold_label] = total_gold.get(gold_label, 0) + 1
        if pred_label == gold_label:
            correct[gold_label] = correct.get(gold_label, 0) + 1
    sorted_items = sorted(total_gold.items(), key=lambda x: x[1], reverse=True)
    logger.info('Results:')
    sum_correct = 0
    sum_pred = 0
    sum_gold = 0
    for k, v in sorted_items:
        sum_correct += correct.get(k, 0)
        sum_pred += total_pred.get(k, 0)
        sum_gold += total_gold.get(k, 0)
        (precision, recall, fscore) = precision_recall_fscore(correct.get(k, 0), total_pred.get(k, 0),
                                                              total_gold.get(k, 0))
        logger.info('{}: {}/{}/{} ({}/{}/{})'.format(vocab.invget(k), precision, recall, fscore, correct.get(k, 0),
                                                     total_pred.get(k, 0), total_gold.get(k, 0)))
    (precision, recall, fscore) = precision_recall_fscore(sum_correct, sum_pred, sum_gold)
    logger.info('Total: {}/{}/{} ({}/{}/{})'.format(precision, recall, fscore, sum_correct, sum_pred, sum_gold))


# def save_model_keras(model, vocab, path):
#     if os.path.isdir(path):
#         os.utime(path, None)
#     else:
#         os.makedirs(path)
#     with open('{}/{}'.format(path, model_filename), 'w') as f:
#         f.write(model.to_json())
#     model.save_weights('{}/{}'.format(path, param_filename), overwrite=True)
#     with open('{}/{}'.format(path, vocab_filename), 'w') as f:
#         f.write(json.dumps(vocab.to_list()))
#
#
# def save_model_sklearn(model, vocab, path):
#     if os.path.isdir(path):
#         os.utime(path, None)
#     else:
#         os.makedirs(path)
#     joblib.dump(model, '{}/{}'.format(path, sklearn_model_filename))
#     with open('{}/{}'.format(path, vocab_filename), 'w') as f:
#         f.write(json.dumps(vocab.to_list()))


def save_model(model, vocab, path):
    if os.path.isdir(path):
        os.utime(path, None)
    else:
        os.makedirs(path)
    with open('{}/{}'.format(path, model_filename), 'w') as f:
        f.write(model.to_json())
    model.save_weights('{}/{}'.format(path, param_filename), overwrite=True)
    with open('{}/{}'.format(path, vocab_filename), 'w') as f:
        f.write(json.dumps(vocab.to_list()))


# def load_model_keras(path):
#     with open('{}/{}'.format(path, model_filename)) as f:
#         model = model_from_json(f.read())
#     model.compile(loss='categorical_crossentropy',
#                   metrics=['accuracy'],
#                   optimizer='adam')
#     model.load_weights('{}/{}'.format(path, param_filename))
#     with open('{}/{}'.format(path, vocab_filename)) as f:
#         vocab = vocabulary.from_list(json.loads(f.read()))
#     return model, vocab
#
#
# def load_model_sklearn(path):
#     model = joblib.load('{}/{}'.format(path, sklearn_model_filename))
#     with open('{}/{}'.format(path, vocab_filename)) as f:
#         vocab = vocabulary.from_list(json.loads(f.read()))
#     return model, vocab


def load_model(path):
    with open('{}/{}'.format(path, model_filename)) as f:
        model = model_from_json(f.read())
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adamax')  # TODO: How to specify the optimizer?
    model.load_weights('{}/{}'.format(path, param_filename))
    with open('{}/{}'.format(path, vocab_filename)) as f:
        vocab = vocabulary.from_list(json.loads(f.read()))
    return model, vocab


######################################################################

def open_data_files(data_file, x_file, y_file):
    if data_file is None:
        if x_file is None:
            raise ValueError('Either data_file or x_file must be specified')
        if y_file is None:
            return {'x': h5py.File(x_file)}
        else:
            return {'x': h5py.File(x_file), 'y': h5py.File(y_file)}
    else:
        return {'xy': h5py.File(data_file)}


def close_data_files(data_files):
    for f in data_files.values():
        f.close()


def load_matrix(h5, path, on_memory=True):
    if csrmatrix.is_csr_file(h5, path):
        # csr matrix is put on memory anyway
        x = csrmatrix.load_csrmatrix(h5, path).todense()
    else:
        if on_memory:
            x = numpy.array(h5[path])
        else:
            x = h5[path]  # directly access hdf5 file
    return x


def load_data(data_files, on_memory=True):
    if 'xy' in data_files:
        x_file = data_files['xy']['x']
        if 'y' in data_files['xy']:
            y_file = data_files['xy']['y']
    else:
        x_file = data_files['x']
        if 'y' in data_files:
            y_file = data_files['y']
    x = load_matrix(x_file, 'data', on_memory)
    if 'index' in x_file:
        index = numpy.array(x_file['index'])
    if y_file is not None:
        # y data is put on memory (should be changed for large vocabulary classification)
        y = load_matrix(y_file, 'data', True)
        if index is None and 'index' in y_file:
            index = numpy.array(y_file['index'])
    else:
        y = None
    if index is None:
        index = numpy.arange(x.shape[0])
    return x, y, index


def load_vocabulary(data_files, vocab_name):
    if 'xy' in data_files:
        return (vocabulary.from_list(data_files['xy']['x'][vocab_name]),
                vocabulary.from_list(data_files['xy']['y'][vocab_name]))
    else:
        assert ('y' in data_files)
        return (vocabulary.from_list(data_files['x'][vocab_name]),
                vocabulary.from_list(data_files['y'][vocab_name]))


def check_data_consistency(x, y, vocab):
    assert (x.shape[0] == y.shape[0])
    assert (y.shape[1] == vocab.size())


def train_mode(args):
    # load training data
    logger.info('Load training data')
    data_files = open_data_files(args.data_file, args.x_file, args.y_file)
    (x, y, index) = load_data(data_files, args.on == 'memory')
    if y is None:
        raise ValueError('y data must be specified for training')
    # x_vocab (question vocabulary), y_vocab (list of candidate answers)
    x_vocab, y_vocab = load_vocabulary(data_files, args.vocab_name)

    if y.shape[1] == 1:
        y = np_utils.to_categorical(y, y_vocab.size())

    check_data_consistency(x, y, y_vocab)

    if args.num_samples > 0:
        (x, y, index) = x[:args.num_samples], y[:args.num_samples], index[:args.num_samples]
    # load validation data
    if args.val_file is not None or (args.val_x_file is not None and args.val_y_file is not None):
        logger.info('Load validation data')
        val_files = open_data_files(args.val_file, args.val_x_file, args.val_y_file)
        (val_x, val_y, val_index) = load_data(val_files, args.on == 'memory')
        assert (val_y is not None)
        if val_y.shape[1] == 1:
            val_y = np_utils.to_categorical(val_y, y_vocab.size())
        check_data_consistency(val_x, val_y, y_vocab)
    if val_x is not None:
        val_data = (val_x, val_y)
    else:
        val_data = None

    logger.info('%s samples, %s labels, %s words', x.shape[0], y.shape[1], x.shape[1])
    check_train_data(x, y)
    #logger.info('Start training: %s', args.model)
    logger.info('Start training:')
    training_start = datetime.now()
    model_params = dict([]) if args.model_params is None else json.loads(args.model_params)
    if args.hyperopt_params is None:
        logger.info('Keras: Train mode')
    else:
        logger.info('Hyperopt + Keras: Search mode')

    model, loss, acc = train(x, y, validation_data=val_data, vocab_size=x_vocab.size(), nb_ans=y_vocab.size(),
                             num_iter=args.num_iter, batch_size=args.batch_size, on_memory=(args.on == 'memory'),
                             early_stopping=args.early_stopping, hyperopt_params=json.loads(args.hyperopt_params),
                             **model_params)
    logger.info('Training time: {}'.format(datetime.now() - training_start))
    logger.info('Loss={}, acc={}'.format(loss, acc))

    close_data_files(data_files)
    if val_data is not None:
        close_data_files(val_files)

    logger.info('Save model: %s', args.model_dir)
    save_model(model, y_vocab, args.model_dir)
    pass


def save_detection_results_hdf5(output_file, pred, index, vocab):
    # TODO: save results in JSONL
    # print(vocab)
    with h5py.File(output_file, 'w') as of:
        pred_array = numpy.array(pred)
        index_array = numpy.array(index)
        vocab_array = numpy.array(vocab.to_list(), dtype=numpy.string_)
        (num_samples, num_labels) = pred.shape
        logger.debug('save detection results for {} samples, {} labels'.format(num_samples, num_labels))
        assert (num_samples == len(index))
        assert (num_labels == vocab.size())
        of['data'] = pred_array
        of['index'] = index_array
        of['columns'] = vocab_array
        of.flush()
    pass


def test_mode(args):
    logger.info('Load model: %s', args.model_dir)
    model, y_vocab = load_model(args.model_dir)

    logger.info('Load test data')
    data_files = open_data_files(args.data_file, args.x_file, args.y_file)
    (test_x, test_y, index) = load_data(data_files, args.on == 'memory')
    if test_y is None:
        test_y = numpy.zeros((len(test_x), y_vocab.size()))
    if test_y.shape[1] == 1:
        test_y = np_utils.to_categorical(test_y, y_vocab.size())
    logger.info('%s samples, %s labels', len(test_x), y_vocab.size())

    logger.info('Start testing')
    val_start = datetime.now()
    check_train_data(test_x, test_y)
    pred_y = model.predict_proba(test_x)
    logger.info('Test time: {}'.format(datetime.now() - val_start))

    close_data_files(data_files)

    show_accuracy(pred_y, test_y, index, y_vocab)
    if args.output_file is not None:
        # save detection results
        save_detection_results_hdf5(args.output_file, pred_y, index, y_vocab)
    pass


######################################################################

if __name__ == '__main__':
    start = datetime.now()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s: %(message)s')

    argparser = argparse.ArgumentParser(description="Train/test Question Bag-of-word classifier")
    argparser.add_argument('action', type=str, choices=['train', 'test'], help='what to do: train or test')
    argparser.add_argument('-d', '--data_file', type=str, dest='data_file',
                           help='training/test data file; the file must contain both x and y')
    argparser.add_argument('-x', '--x_file', type=str, dest='x_file', help='feature data file (i.e. x)')
    argparser.add_argument('-y', '--y_file', type=str, dest='y_file', help='label data file (i.e. y)')
    argparser.add_argument('--val', type=str, dest='val_file',
                           help='validation data file; the file must contain both x and y')
    argparser.add_argument('--val_x', type=str, dest='val_x_file', help='feature data file for validation')
    argparser.add_argument('--val_y', type=str, dest='val_y_file', help='label data file for validation')
    argparser.add_argument('-f', '--model_dir', type=str, dest='model_dir', help='directory to store the model')
    argparser.add_argument('-o', '--output_file', type=str, dest='output_file', default=None,
                           help='file to output detection results')
    argparser.add_argument('-i', '--num_iter', type=int, dest='num_iter', default=10,
                           help='number of epochs for training')
    argparser.add_argument('-n', '--num_samples', type=int, dest='num_samples', default=0,
                           help='number of training samples')
    argparser.add_argument('--vocab_name', type=str, dest='vocab_name', default='columns', help='vocabulary of y')
    argparser.add_argument('--batch_size', type=int, dest='batch_size', default=1,
                           help='Batch size for training/testing')
    argparser.add_argument('--on', type=str, dest='on', default='memory', choices=['memory', 'file'],
                           help='put data on memory or file')
    argparser.add_argument('--early_stopping', type=int, dest='early_stopping', default=-1,
                           help='patience for early stopping (negative value means no early stopping)')
    argparser.add_argument('--model_params', type=str, dest='model_params', default=None,
                           help='dict (JSON format) of hyperparameters passed to the model')
    argparser.add_argument('--hyperopt_params', type=str, dest='hyperopt_params', default=None,
                           help='dict (JSON format) of hyperparameter candidates to be optimized with hyperopt; hyperparameter optimization turned off if None')
    args = argparser.parse_args()
    logger.info('Command-line arguments: %s', args)

    if args.action == 'train':
        train_mode(args)
    else:
        assert (args.action == 'test')
        test_mode(args)

    logger.info('done')
    logger.info('Time: {}'.format(datetime.now() - start))
