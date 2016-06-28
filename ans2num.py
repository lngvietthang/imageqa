# coding: utf-8
"""convert input answers into numerical representations"""

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


def build_dict(lst_txt, padding=True):
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
    key_start = 0
    if padding:
        key_start = 1
        word_dict['padding'] = 0
    word_dict['UNK'] = key_start
    for ii, ww in enumerate(sorted_words):
        word_dict[ww] = ii + 1 + key_start

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

        argparser = argparse.ArgumentParser(description="Make numerical representations of answers")
        argparser.add_argument("answer_file", type=str, help="JSONL file of texts")
        argparser.add_argument("output_file", type=str, help="hdf5 file to output matrix")
        argparser.add_argument("--vocab_file", type=str,
                               help="vocabulary file")
        argparser.add_argument("--ans_attr", type=str, default='answer', help="attribute name for answer in input JSONL")
        argparser.add_argument("--qid_attr", type=str, default='question_id', help="attribute name for question ID in input JSONL")
        argparser.add_argument("--data_attr", type=str, default='data',
                               help="attribute to store list of numbers in output HDF5")
        argparser.add_argument("--index_attr", type=str, default='index',
                               help="attribute name for index in output HDF5")

        args = argparser.parse_args()

        logger.info("Read texts and compute text vectors")
        lst_index = []
        lst_ans = []
        with open(args.answer_file) as input:
            for line in input:
                js = json.loads(line)
                quest_id = js[args.qid_attr]
                lst_index.append(quest_id)
                quest = preprocess(js[args.ans_attr])
                lst_ans.append(quest)
        lst_ans = [['_'.join([word for word in ans])] for ans in lst_ans]  # multiple words in ans

        if args.vocab_file is None:
            logger.info("Build vocabulary")
            word_dict = build_dict(lst_ans, padding=False)
        else:
            logger.info("Load vocabulary")
            word_dict = load_dict(args.vocab_file)

        logger.info("Convert text to number")
        arr_ans = []
        for ans in lst_ans:
            if ans[0] in word_dict:
                arr_ans.append(word_dict[ans[0]])
            else:
                arr_ans.append(word_dict['UNK'])
        arr_ans = np.array(arr_ans).reshape(len(arr_ans), 1)

        logger.info("Output numerical representations of answer")
        with h5py.File(args.output_file, "w") as output:
            output.create_dataset(args.index_attr, data=np.array(lst_index))
            output.create_dataset(args.data_attr, data=arr_ans)
            columns = [word.encode('utf8') for word in word_dict.keys()]
            output.create_dataset("columns", data=columns)

        if args.vocab_file is None:
            logger.info("Save vocabulary")
            path = os.path.join(os.path.splitext(args.output_file)[0] + '.dict')
            save_dict(path, word_dict)

        logger.info("done")