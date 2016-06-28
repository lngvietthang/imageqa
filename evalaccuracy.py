# coding: utf-8

import sys

reload(sys)
sys.setdefaultencoding("utf-8")

import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import json
import h5py
import argparse
import numpy
import logging
from utils import vocabulary

logger = logging.getLogger(__name__)

def load_gold_answers(filename):
    with open(filename) as f:
        return { e['question_id']: e['answers'] for e in [json.loads(line) for line in f] }

def load_answers_jsonl(filename):
    with open(filename) as f:
        return { e['question_id']: e['answer'] for e in [json.loads(line) for line in f] }

def load_answers_hdf5(filename):
    with h5py.File(filename) as f:
        index = f['index']
        vocab = vocabulary.from_list(f['columns'])
        data = f['data']
        answers = [vocab.invget(numpy.argmax(v)) for v in data]
        return dict(zip(index, answers))

def evaluate(gold, pred, eval_file):
    num_correct = 0
    eval_file.write('id\tpred\tgold\n')
    ids = sorted(pred.keys())
    for qid in ids:
        answer = pred[qid]
        gold_answers = gold[qid]
        if answer in gold_answers:
            num_correct += 1
        eval_file.write('{}\t{}\t{}\n'.format(qid, answer, ','.join(gold_answers)))
    num_gold = len(gold)
    num_pred = len(pred)
    logger.debug('correct, gold, pred = %s, %s, %s', num_correct, num_gold, num_pred)
    precision = float(num_correct) / num_pred
    recall = float(num_correct) / num_gold
    if (precision + recall) == 0:
        fscore = 0.0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s: %(message)s')

    argparser = argparse.ArgumentParser(description = "evaluate image QA accuracy")
    argparser.add_argument("-g", dest="gold_json", required=True, type=str, help="JSONL file of gold answers")
    argparser.add_argument("-p", dest="pred_file", required=True, type=str, help="JSONL or HDF5 file of predicted answers")
    argparser.add_argument("-l", dest="log_file", required=True, type=str, help="text file to record evaluation result")
    args = argparser.parse_args()
    logger.debug(args)

    logger.debug('load gold answers: %s', args.gold_json)
    gold_answers = load_gold_answers(args.gold_json)
    if args.pred_file.endswith('.h5'):
        logger.debug('load pred answers from hdf5: %s', args.pred_file)
        pred_answers = load_answers_hdf5(args.pred_file)
    else:
        logger.debug('load pred answers from jsonl: %s', args.pred_file)
        assert(args.pred_file.endswith('.jsonl'))
        pred_answers = load_answers_jsonl(args.pred_file)
    logger.debug('compute accuracy')
    with open(args.log_file, 'w') as f:
        precision, recall, fscore = evaluate(gold_answers, pred_answers, f)
        f.write('{}\t{}\t{}\t{}\n'.format(args.log_file, precision, recall, fscore))

