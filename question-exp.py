# coding: utf-8
"""Run experiments on Image QA only using Question"""

from __future__ import print_function, unicode_literals
import sys
import logging
import json
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

from core.imageqa.runexp import Workflow

logger = logging.getLogger(__name__)

## Tasks to run cocoqa experiments

# root directory of stable data
data_root = '../../resources/data'
# root directory of experimental data
exp_root = '../../resources/experiment/question-only'

question_max_len = [30, 45, 50]

datasets = ['cocoqa2015']

dataset_specs = {
    'cocoqa2015': {
        'directory': 'cocoqa-2015-05-17',
        'splits': ['dev1', 'dev2', 'val'],
        'feature_source': 'mscoco2014'
    },
    'vqa2015oe': {
        'directory': 'VQA_2015_v1.0',
        'splits': ['dev1', 'dev2', 'val'],
        'feature_source': 'mscoco2014',
        'answer_prefix': 'vqa2015',
    },
    'vqa2015mc': {
        'directory': 'VQA_2015_v1.0',
        'splits': ['dev1', 'dev2', 'val'],
        'feature_source': 'mscoco2014',
        'answer_prefix': 'vqa2015',
    },
    'daquarfull': {
        'directory': 'DAQUAR',
        'splits': ['dev1', 'dev2', 'test'],
        'feature_source': 'daquar'
    },
    'daquar': {
        'directory': 'DAQUAR',
        'splits': ['dev1', 'dev2', 'val']
    }
}

train_methods = {
    "bow": {
        "model": "mlp",
        "train_args": "-i 1000 --batch_size 100 --early_stopping 10 --hyperopt_params '{}'".format(json.dumps(
            {'wembdim': [100, 200, 300, 500],
             'wembinit': ['uniform', 'normal', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform'],
             'wembdropout': [0.0, 0.1, 0.3, 0.5],
             'optimizer': ['adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax'],
             'maxevals': 15})),
        "test_args": "--batch_size 100"
    },
}


def make_exp_dir(task):
    task(name='make dir: {}'.format(exp_root),
         rule='mkdir -p {}'.format(exp_root),
         target=exp_root)


def build_question_numerical(task):
    """build numerical representations for questions"""
    for dataset_name in datasets:
        dataset_spec = dataset_specs[dataset_name]
        dataset_dir = data_root + '/' + dataset_spec['directory']
        # Process dev1 first to get the vocabulary
        quest_file = '{}/{}_dev1_questions.jsonl'.format(dataset_dir, dataset_name)
        for max_len in question_max_len:
            output_file = '{}/{}_dev1_questions_maxlen{}.h5'.format(exp_root, dataset_name, max_len)
            task(name='Q numerical representations: {}_dev1-maxlen{}'.format(dataset_name, max_len),
                 rule='python quest2num.py {} {} --max_len {} --quest_attr question --qid_attr question_id'.format(
                     quest_file, output_file, max_len),
                 source=quest_file,
                 target=output_file)
        # Process dev2 and val(test) set
        for data_split in dataset_spec['splits']:
            if data_split == 'dev1':
                continue
            quest_file = '{}/{}_{}_questions.jsonl'.format(dataset_dir, dataset_name, data_split)
            for max_len in question_max_len:
                output_file = '{}/{}_{}_questions_maxlen{}.h5'.format(exp_root, dataset_name, data_split, max_len)
                vocab_file = '{}/{}_dev1_questions_maxlen{}.dict'.format(exp_root, dataset_name, max_len)
                task(name='Q numerical representations: {}_{}-maxlen{}'.format(dataset_name, data_split, max_len),
                     rule='python quest2num.py {} {} --vocab_file {} --max_len {} --quest_attr question --qid_attr question_id'.format(
                         quest_file, output_file, vocab_file, max_len),
                     source=quest_file,
                     target=output_file)

    pass


def build_answer_numerical(task):
    """build numerical representations for answers"""
    for dataset_name in datasets:
        dataset_spec = dataset_specs[dataset_name]
        dataset_dir = data_root + '/' + dataset_spec['directory']
        # Process dev1 first to get the vocabulary
        ans_file = '{}/{}_dev1_answers.jsonl'.format(dataset_dir, dataset_name)
        output_file = '{}/{}_dev1_answers.h5'.format(exp_root, dataset_name)
        task(name='A numerical representations: {}_dev1'.format(dataset_name),
             rule='python ans2num.py {} {} --ans_attr answer --qid_attr question_id'.format(
                 ans_file, output_file),
             source=ans_file,
             target=output_file)
        # Process dev2 and val(test) set
        vocab_file = '{}/{}_dev1_answers.dict'.format(exp_root, dataset_name)
        for data_split in dataset_spec['splits']:
            if data_split == 'dev1':
                continue
            ans_file = '{}/{}_{}_answers.jsonl'.format(dataset_dir, dataset_name, data_split)
            output_file = '{}/{}_{}_answers.h5'.format(exp_root, dataset_name, data_split)
            task(name='Q numerical representations: {}_{}'.format(dataset_name, data_split),
                 rule='python ans2num.py {} {} --vocab_file {} --ans_attr answer --qid_attr question_id'.format(
                     ans_file, output_file, vocab_file),
                 source=ans_file,
                 target=output_file)

    pass


def train_and_eval(task):
    for dataset in datasets:
        dataset_spec = dataset_specs[dataset]
        dataset_dir = data_root + '/' + dataset_spec['directory']
        for max_len in question_max_len:
            train_x = '{}/{}_dev1_questions_maxlen{}.h5'.format(exp_root, dataset, max_len)
            train_y = '{}/{}_dev1_answers.h5'.format(exp_root, dataset)

            val_x = '{}/{}_dev2_questions_maxlen{}.h5'.format(exp_root, dataset, max_len)
            val_y = '{}/{}_dev2_answers.h5'.format(exp_root, dataset)

            test_x = '{}/{}_val_questions_maxlen{}.h5'.format(exp_root, dataset, max_len)
            test_y = '{}/{}_val_answers.h5'.format(exp_root, dataset)
            test_gold = '{}/{}_val_answers.jsonl'.format(dataset_dir, dataset)

            for train_method in train_methods.keys():
                model_file = '{}/{}_dev1_model-{}_maxlen{}'.format(exp_root, dataset, train_method, max_len)
                test_pred = '{}/{}_val_predictions-{}_maxlen{}.h5'.format(exp_root, dataset, train_method, max_len)
                test_eval = '{}/{}_val_evalresult-{}_maxlen{}.txt'.format(exp_root, dataset, train_method, max_len)
                # train and validation
                train_params = train_methods[train_method]['train_args']
                task(name='train {}: {}_{}'.format(train_method, dataset, max_len),
                     source=[train_x, train_y, val_x, val_y],
                     target=model_file,
                     rule='python bow.py train -x {} -y {} -f {} --val_x {} --val_y {} {}'.format(
                         train_x, train_y, model_file, val_x, val_y, train_params))

                # predict on test data
                test_params = train_methods[train_method]['test_args']
                task(name='predict {}: {}_maxlen{}'.format(train_method, dataset, max_len),
                     source=[test_x, test_y, model_file],
                     target=test_pred,
                     rule='python bow.py test -x {} -y {} -f {} -o {} {}'.format(test_x, test_y, model_file, test_pred,
                                                                                 test_params))
                # evaluate on test data
                task(name='evaluate {}: {}_maxlen{}'.format(train_method, dataset, max_len),
                     source=[test_pred, test_gold],
                     target=[test_eval],
                     rule='python evalaccuracy.py -g {} -p {} -l {}'.format(test_gold, test_pred, test_eval))

    pass


######################################################################

## Main

logger.debug('set tasks')
exp = Workflow()
make_exp_dir(exp)
build_question_numerical(exp)
build_answer_numerical(exp)
train_and_eval(exp)

exp.set_options(environments_distributed=[{'THEANO_FLAGS': 'device=gpu{}'.format(i)} for i in range(exp.num_jobs)])

logger.debug('run tasks')
if not exp.run():
    sys.exit(1)

