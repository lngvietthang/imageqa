# coding: utf-8
"""Train Question Bag-of-word classifier"""

from __future__ import print_function
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../utils')

import argparse
import numpy
import scipy
import logging
from datetime import datetime
import h5py
import json
import itertools
import tempfile

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.externals import joblib
# from sklearn.cross_validation import train_test_split
# from sklearn.grid_search import GridSearchCV
# #from sknn.mlp import Classifier, Layer
from sklearn.preprocessing import label_binarize

import vocabulary
import csrmatrix

logger = logging.getLogger(__name__)

model_filename = 'model.json'
param_filename = 'param.h5'
sklearn_model_filename = 'model.pkl'
vocab_filename = 'vocab.json'

######################################################################
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


def mlp(input_size, output_size, num_units=1024, activation='relu', dropout=0.5, l2=0.0, optimizer='adam'):
    logger.info(
        'num_units={}, activation={}, dropout={}, l2={}, optimizer={}'.format(num_units, activation, dropout, l2,
                                                                              optimizer))
    reg_l2 = regularizers.l2(l2) if l2 > 0.0 else None
    model = Sequential()
    model.add(Dense(num_units, input_dim=input_size, activation=activation, W_regularizer=reg_l2))
    # model.add(BatchNormalization(input_shape=(input_size,), mode=0))
    # model.add(Dense(num_units, activation=activation, W_regularizer=reg_l2))
    if dropout > 0.0: model.add(Dropout(dropout))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax', W_regularizer=reg_l2))
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=optimizer)
    return model


def loglinear(input_size, output_size, l2=0.0, optimizer='adam'):
    logger.info('l2={}, optimizer={}'.format(l2, optimizer))
    reg_l2 = regularizers.l2(l2) if l2 > 0.0 else None
    model = Sequential()
    model.add(Dense(output_size, input_dim=input_size, activation='softmax', W_regularizer=reg_l2))
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=optimizer)
    return model


# callback to flush stdout - in order to see log messages timely
class FlushStdout(Callback):
    def on_batch_begin(self, batch, logs={}):
        sys.stdout.flush()

    def on_train_end(self, logs={}):
        sys.stdout.flush()


def train_keras(x, y, model_name, validation_data, num_iter, batch_size=1, on_memory=True, early_stopping=-1,
                num_jobs=1, **model_params):
    num_labels = y.shape[1]
    feature_size = x.shape[1]
    if model_name == 'loglinear':
        model = loglinear(feature_size, num_labels, **model_params)
    else:
        assert (model_name == 'mlp')
        model = mlp(feature_size, num_labels, **model_params)
    shuffle = True if on_memory else 'batch'
    callbacks = [EarlyStopping(monitor='val_loss', patience=early_stopping)] if early_stopping >= 0 else []
    temp_model = tempfile.NamedTemporaryFile()
    callbacks.append(ModelCheckpoint(filepath=temp_model.name, save_best_only=True))
    callbacks.append(FlushStdout())
    history = model.fit(x, y, validation_data=validation_data, nb_epoch=num_iter, batch_size=batch_size, verbose=2,
                        shuffle=shuffle, callbacks=callbacks)
    # print(history.history)
    sys.stdout.flush()
    model.load_weights(temp_model.name)
    if validation_data is not None:
        loss, acc = model.evaluate(validation_data[0], validation_data[1], batch_size=batch_size)
    else:
        loss, acc = 0.0, 0.0
    temp_model.close()
    return model, loss, acc


def onehot2label(mat):
    """convert one-hot representation into integer labels"""
    labels = [numpy.argmax(l) for l in mat]
    return labels


def train_sklearn(x, y, model_name, validation_data, num_iter, on_memory=True,
                  num_jobs=1, l2=0.0001):
    assert (model_name == 'loglinear')
    assert (l2 > 0.0)
    logger.info('l2={}'.format(l2))
    assert (len(y.shape) == 2)
    model = OneVsRestClassifier(
        SGDClassifier(loss='log', penalty='l2', alpha=l2, n_iter=num_iter, verbose=2))  # , n_jobs=num_jobs)
    # model = LogisticRegression(penalty='l2', C=l2, max_iter=num_iter, multi_class='multinomial', solver='lbfgs', verbose=2)
    model.fit(x, y)
    pred_y = model.predict_proba(validation_data[0])
    val_loss = log_loss(validation_data[1], pred_y)
    val_acc = accuracy_score(onehot2label(validation_data[1]), onehot2label(pred_y))
    return model, val_loss, val_acc  # TODO: return val_loss and val_acc


def generate_params(grid):
    if grid is None: return [{}]
    assert (isinstance(grid, dict))
    keys = grid.keys()
    values = grid.values()
    return [dict(zip(keys, params)) for params in itertools.product(*values)]


# def train(x, y, model_name, validation_data, num_iter, batch_size=1, on_memory=True, early_stopping=-1, num_jobs=1,
#           framework='keras',
#           param_grid=None, **model_params):
def train(x, y, model_name, validation_data, num_iter, batch_size=1, on_memory=True, early_stopping=-1, hyperopt_params,
          **model_params):
    best_model = None
    best_loss = None
    best_acc = None
    # best_params = None
    # logs = []
    # params_list = generate_params(param_grid)
    # logger.info('Start training: {} parameter grids'.format(len(params_list)))
    # for i, params in enumerate(params_list):
    #     logger.info('Grid {}'.format(i + 1))
    #     merged_params = dict(model_params, **params)
    #     if framework == 'keras':
    #         model, loss, acc = train_keras(x, y, model_name, validation_data, num_iter,
    #                                        batch_size=batch_size, on_memory=on_memory, early_stopping=early_stopping,
    #                                        num_jobs=num_jobs,
    #                                        **merged_params)
    #     else:
    #         assert (framework == 'sklearn')
    #         model, loss, acc = train_sklearn(x, y, model_name, validation_data, num_iter,
    #                                          on_memory=on_memory, num_jobs=num_jobs,
    #                                          **merged_params)
    #     # if best_model is None or acc > best_acc:
    #     if best_model is None or loss < best_loss:
    #         best_model, best_loss, best_acc, best_params = model, loss, acc, params
    #     logs.append(params.values() + [loss, acc])
    #     logger.info('Grid {} done: loss={}, acc={}'.format(i + 1, loss, acc))
    # logger.info('Best loss={}, acc={}, model: {}'.format(best_loss, best_acc, best_params))
    # logger.info('{}\tloss\tacc'.format('\t'.join(params.keys())))
    # for log in logs:
    #     logger.info('\t'.join([str(l) for l in log]))

    def keras_fmin_fnct(space):
        """

        :param space:
        :return:
        """
        quest_model = Sequential()
        quest_model.add(Embedding(input_dim=vocab_size, output_dim=space['WEmbDim'],
                                  init=space['WEmbInit'],
                                  mask_zero=False, dropout=space['WEmbDrop']
                                  )
                        )
        quest_model.add(Lambda(function=lambda x: K.sum(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:]))
        quest_model.add(Dense(nb_ans))
        quest_model.add(Activation('softmax'))

        quest_model.compile(loss='categorical_crossentropy',
                            optimizer={{choice(['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax'])}},
                            metrics=['accuracy'])

        print('##################################')
        print('Train...')
        callbacks = [EarlyStopping(monitor='val_loss', patience=early_stopping)] if early_stopping >= 0 else []
        temp_model = tempfile.NamedTemporaryFile()
        callbacks.append(ModelCheckpoint(filepath=temp_model.name, save_best_only=True))
        callbacks.append(FlushStdout())

        quest_model.fit(x, y, batch_size=batch_size, nb_epoch=num_iter,
                        validation_data=validation_data,
                        callbacks=callbacks)

        score, acc = quest_model.evaluate(validation_data[0], validation_data[1], verbose=1)

        print('##################################')
        print('Test accuracy:%.4f' % acc)

        return {'loss': -acc, 'status': STATUS_OK, 'model': quest_model}

    def get_space():
        return {
            'WEmbDim': hp.choice('WEmbDim', hyperopt_params['wembdim']),
            'WEmbInit': hp.choice('WEmbInit', hyperopt_params['wembinit']),
            'WEmbDrop': hp.choice('WEmbDrop', hyperopt_params['wembdropout']),
        }


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
    return (precision, recall, fscore)


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
        if pred_label == gold_label: correct[gold_label] = correct.get(gold_label, 0) + 1
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


def save_model_keras(model, vocab, path):
    if os.path.isdir(path):
        os.utime(path, None)
    else:
        os.makedirs(path)
    with open('{}/{}'.format(path, model_filename), 'w') as f:
        f.write(model.to_json())
    model.save_weights('{}/{}'.format(path, param_filename), overwrite=True)
    with open('{}/{}'.format(path, vocab_filename), 'w') as f:
        f.write(json.dumps(vocab.to_list()))


def save_model_sklearn(model, vocab, path):
    if os.path.isdir(path):
        os.utime(path, None)
    else:
        os.makedirs(path)
    joblib.dump(model, '{}/{}'.format(path, sklearn_model_filename))
    with open('{}/{}'.format(path, vocab_filename), 'w') as f:
        f.write(json.dumps(vocab.to_list()))


def save_model(model, vocab, path, framework):
    if framework == 'keras':
        save_model_keras(model, vocab, path)
    else:
        assert (framework == 'sklearn')
        save_model_sklearn(model, vocab, path)


def load_model_keras(path):
    with open('{}/{}'.format(path, model_filename)) as f:
        model = model_from_json(f.read())
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam')
    model.load_weights('{}/{}'.format(path, param_filename))
    with open('{}/{}'.format(path, vocab_filename)) as f:
        vocab = vocabulary.from_list(json.loads(f.read()))
    return model, vocab


def load_model_sklearn(path):
    model = joblib.load('{}/{}'.format(path, sklearn_model_filename))
    with open('{}/{}'.format(path, vocab_filename)) as f:
        vocab = vocabulary.from_list(json.loads(f.read()))
    return model, vocab


def load_model(path, framework):
    if framework == 'keras':
        return load_model_keras(path)
    else:
        assert (framework == 'sklearn')
        return load_model_sklearn(path)


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
        y = load_matrix(y_file, 'data',
                        True)  # y data is put on memory (should be changed for large vocabulary classification)
        if index is None and 'index' in y_file:
            index = numpy.array(y_file['index'])
    else:
        y = None
    if index is None:
        index = numpy.arange(x.shape[0])
    return x, y, index


def load_vocabulary(data_files, vocab_name):
    if 'xy' in data_files:
        return vocabulary.from_list(data_files['xy']['y'][vocab_name])
    else:
        assert ('y' in data_files)
        return vocabulary.from_list(data_files['y'][vocab_name])


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
    vocab = load_vocabulary(data_files, args.y_vocab)
    if len(y.shape) == 1:
        y = label_binarize(y, classes=range(vocab.size()))
    check_data_consistency(x, y, vocab)
    if args.num_samples > 0:
        (x, y, index) = x[:args.num_samples], y[:args.num_samples], index[:args.num_samples]
    # load validation data
    if args.val_file is not None or (args.val_x_file is not None and args.val_y_file is not None):
        logger.info('Load validation data')
        val_files = open_data_files(args.val_file, args.val_x_file, args.val_y_file)
        (val_x, val_y, val_index) = load_data(val_files, args.on == 'memory')
        assert (val_y is not None)
        if len(val_y.shape) == 1:
            val_y = label_binarize(val_y, classes=range(vocab.size()))
        check_data_consistency(val_x, val_y, vocab)
    if val_x is not None:
        val_data = (val_x, val_y)
    else:
        val_data = None

    logger.info('%s samples, %s labels, %s features', x.shape[0], y.shape[1], x.shape[1])
    check_train_data(x, y)
    logger.info('Start training: %s', args.model)
    training_start = datetime.now()
    model_params = dict([]) if args.model_params is None else json.loads(args.model_params)
    if args.hyperopt_params is None:
        logger.info('Keras: Train mode')
    else:
        logger.info('Hyperopt + Keras: Search mode')
    # model, loss, acc = train(x, y, model_name=args.model, validation_data=val_data, num_iter=args.num_iter,
    #                          batch_size=args.batch_size, on_memory=(args.on == 'memory'),
    #                          early_stopping=args.early_stopping, num_jobs=args.num_jobs, framework=args.framework,
    #                          param_grid=json.loads(args.param_grid), **model_params)

    model, loss, acc = train(x, y, model_name=args.model, validation_data=val_data, num_iter=args.num_iter,
                             batch_size=args.batch_size, on_memory=(args.on == 'memory'),
                             early_stopping=args.early_stopping, hyperopt_params=json.loads(args.hyperopt_params), **model_params)
    logger.info('Training time: {}'.format(datetime.now() - training_start))
    logger.info('Loss={}, acc={}'.format(loss, acc))

    close_data_files(data_files)
    if val_data is not None:
        close_data_files(val_files)

    # if len(val_y) > 0:
    #     logger.info('Validation')
    #     val_start = datetime.now()
    #     check_train_data(val_x, val_y)
    #     score = model.evaluate(val_x, val_y, batch_size=16)
    #     logger.info('Validation time: {}'.format(datetime.now() - val_start))
    #     logger.info('Accuracy: {}'.format(score))

    logger.info('Save model: %s', args.model_dir)
    save_model(model, vocab, args.model_dir, args.framework)
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
    model, vocab = load_model(args.model_dir, args.framework)

    logger.info('Load test data')
    data_files = open_data_files(args.data_file, args.x_file, args.y_file)
    (test_x, test_y, index) = load_data(data_files, args.on == 'memory')
    if test_y is None:
        test_y = numpy.zeros((len(test_x), vocab.size()))
    if len(test_y.shape) == 1:
        test_y = label_binarize(test_y, classes=range(vocab.size()))
    logger.info('%s samples, %s labels', len(test_x), vocab.size())

    logger.info('Start testing')
    val_start = datetime.now()
    check_train_data(test_x, test_y)
    # pred_y = model.predict_proba(test_x, batch_size=args.batch_size)
    pred_y = model.predict_proba(test_x)
    logger.info('Test time: {}'.format(datetime.now() - val_start))

    close_data_files(data_files)

    show_accuracy(pred_y, test_y, index, vocab)
    if args.output_file is not None:
        # save detection results
        save_detection_results_hdf5(args.output_file, pred_y, index, vocab)
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
    # argparser.add_argument('-l', '--log_file', type=str, dest='log_file', default=None, help='file to output log messages')
    # argparser.add_argument('-m', '--model', type=str, dest='model', default='loglinear', choices=['loglinear', 'mlp'],
    #                       help='model of classifier')
    # argparser.add_argument('--framework', type=str, dest='framework', default='keras', choices=['keras', 'sklearn'],
    #                       help='machine learning framework to be used')
    argparser.add_argument('-i', '--num_iter', type=int, dest='num_iter', default=10,
                           help='number of epochs for training')
    # argparser.add_argument('-j', '--num_jobs', type=int, dest='num_jobs', default=1, help='number of parallel jobs')
    # argparser.add_argument('-s', '--split_ratio', type=float, dest='split_ratio', default=0.0,
    #                       help='ratio of train/val split')
    argparser.add_argument('-n', '--num_samples', type=int, dest='num_samples', default=0,
                           help='number of training samples')
    argparser.add_argument('--y_vocab', type=str, dest='y_vocab', default='columns', help='vocabulary of y')
    # argparser.add_argument('-p', '--threshold', type=float, dest='threshold', default=0.5, help='probability threshold to output label')
    # argparser.add_argument('--num_units', type=int, dest='num_units', default=100, help='Number of units of hidden layer (multilayer perceptron only)')
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