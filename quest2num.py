# coding: utf-8
"""convert input question into numerical representations (for Keras Embedding layer)"""

from __future__ import print_function
import argparse
import logging
import json
import pickle
import os
import h5py
from collections import OrderedDict

from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np

logger = logging.getLogger(__name__)


def preprocess(line, is_lmz=False):
    line = wordpunct_tokenize(line.strip())
    if is_lmz:
        lemmatizer = WordNetLemmatizer()
        line = [lemmatizer.lemmatize(word) for word in line]

    return line


def build_dict(lst_txt):
    word_freqs = OrderedDict()
    for line in lst_txt:
        #        words_in = line.strip().split(' ')
        #        for w in words_in:
        for w in line:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1
    words = word_freqs.keys()
    freqs = word_freqs.values()

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    word_dict = OrderedDict()
    word_dict['padding'] = 0
    word_dict['UNK'] = 1
    for ii, ww in enumerate(sorted_words):
        word_dict[ww] = ii + 2

    return word_dict


def save_dict(path, word_dict):
    with open(path, 'w') as fwrite:
        pickle.dump(word_dict, fwrite)
    pass


def load_dict(path):
    with open(path) as fread:
        word_dict = pickle.load(fread)
    return word_dict


if __name__ == '__main__':
    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s: %(message)s')

        argparser = argparse.ArgumentParser(description="Make numerical representations of texts")
        argparser.add_argument("question_file", type=str, help="JSONL file of texts")
        argparser.add_argument("output_file", type=str, help="hdf5 file to output matrix")
        argparser.add_argument("--vocab_file", type=str,
                               help="vocabulary file")
        argparser.add_argument("--max_len", type=int, default=30,
                               help="maximum length of a question")
        argparser.add_argument("--quest_attr", type=str, default='question', help="attribute name for questions in input JSONL")
        argparser.add_argument("--qid_attr", type=str, default='question_id', help="attribute name for question ID in input JSONL")
        argparser.add_argument("--data_attr", type=str, default='data',
                               help="attribute to store list of numbers in output HDF5")
        argparser.add_argument("--index_attr", type=str, default='index',
                               help="attribute name for index in output HDF5")

        args = argparser.parse_args()

        logger.info("Read texts and compute text vectors")
        lst_index = []
        lst_quests = []
        with open(args.question_file) as input:
            for line in input:
                js = json.loads(line)
                quest_id = js[args.qid_attr]
                lst_index.append(quest_id)
                quest = preprocess(js[args.quest_attr])
                lst_quests.append(quest)

        if args.vocab_file is None:
            logger.info("Build vocabulary")
            word_dict = build_dict(lst_quests)
        else:
            logger.info("Load vocabulary")
            word_dict = load_dict(args.vocab_file)

        logger.info("Convert text to number")
        nb_ins = len(lst_quests)
        max_len = args.max_len
        arr_quests = np.zeros((nb_ins, max_len), dtype=int)
        for i in range(nb_ins):
            for j in range(min(max_len, len(lst_quests[i]))):
                if lst_quests[i][j] in word_dict:
                    arr_quests[i, j] = word_dict[lst_quests[i][j]]
                else:
                    arr_quests[i, j] = word_dict['UNK']

        with h5py.File(args.output_file, "w") as output:
            output.create_dataset(args.index_attr, data=np.array(lst_index))
            output.create_dataset(args.data_attr, data=arr_quests)
            output.create_dataset("columns", data=word_dict.keys())

        if args.vocab_file is None:
            logger.info("Save vocabulary")
            path = os.path.join(os.path.dirname(args.output_file), os.path.splitext(args.output_file)[0] + '.dict')
            save_dict(path, word_dict)

        logger.info("done")