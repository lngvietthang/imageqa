import json
import h5py
import os
import argparse
from collections import OrderedDict

import numpy as np
import pickle
from scipy import sparse

from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer

verbose = os.environ.get('VERBOSE', 'no') == 'yes'
debug = os.environ.get('DEBUG', 'no') == 'yes'


def preprocess(lines, is_lmz=True):
    lines = [wordpunct_tokenize(line.strip()) for line in lines]
    if is_lmz:
        lemmatizer = WordNetLemmatizer()
        lines = [[lemmatizer.lemmatize(word) for word in line] for line in lines]

    return lines


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


def conv_txtdata2num(lst_quest, lst_ans, maxlen):
    nb_ins = len(lst_quest)
    qword_dict = build_dict(lst_quest)
    vocab_size = len(qword_dict)
    q = np.zeros((nb_ins, maxlen), dtype=int)
    for i in range(nb_ins):
        for j in range(min(maxlen, len(lst_quest[i]))):
            if lst_quest[i][j] in qword_dict:
                q[i, j] = qword_dict[lst_quest[i][j]]
            else:
                q[i, j] = qword_dict['UNK']

    lst_ans = [['_'.join([word for word in ans])] for ans in lst_ans]  # multiple words in ans
    aword_dict = build_dict(lst_ans)
    nb_ans = len(aword_dict) - 1
    a = []
    for ans in lst_ans:
        if ans[0] in aword_dict:
            a.append(aword_dict[ans[0]] - 1)
        else:
            a.append(aword_dict['UNK'] - 1)
    a = np.array(a).reshape(len(a), 1)

    # Verbose
    if verbose:
        print 'Question Vocabulary Size:', vocab_size
        print 'Number of candidate answers:', nb_ans

    return (q, a, qword_dict, aword_dict)


def conv_txtdata2num_withdict(lst_quest, lst_ans, maxlen, qword_dict, aword_dict):
    nb_ins = len(lst_quest)
    q = np.zeros((nb_ins, maxlen), dtype=int)
    for i in range(nb_ins):
        for j in range(min(maxlen, len(lst_quest[i]))):
            if lst_quest[i][j] in qword_dict:
                q[i, j] = qword_dict[lst_quest[i][j]]
            else:
                q[i, j] = qword_dict['UNK']

    lst_ans = [['_'.join([word for word in ans]) for ans in lst_ans]]  # multiple words in ans
    a = []
    for ans in lst_ans:
        if ans[0] in aword_dict:
            a.append(aword_dict[ans[0]] - 1)
        else:
            a.append(aword_dict['UNK'] - 1)
    a = np.array(a).reshape(len(a), 1)
    return q, a


def map_imgid2idx(lst_imgid, lst_imgidx):
    result = np.zeros((len(lst_imgid), 1), dtype=int)
    for i, ins in enumerate(lst_imgid):
        idx = np.where(lst_imgidx == ins)
        if len(idx) == 1:
            result[i] = idx[0]
        else:
            print "[1] Problem in the list of image ids."
            raise ValueError('[1] Problem in the list of image ids')
    return result


def load_imgfeat(path2datadir, datapart):
    path2imgfeat = os.path.join(path2datadir, 'mscoco2014_' + datapart + '_vgg-fc7.h5')
    fdata = h5py.File(path2imgfeat)
    data = fdata['data'][:]
    index = fdata['index'][:]

    return data, index


def load_data(path2datadir, dataname, datapart):
    path2q = os.path.join(path2datadir, dataname + '_' + datapart + '_questions.json')
    path2a = os.path.join(path2datadir, dataname + '_' + datapart + '_answers.json')

    # Load Training Questions and Answers
    lstq = []
    lsta = []
    lsti = []
    with open(path2q) as fread:
        for line in fread:
            json_str = json.loads(line)
            lstq.append(json_str['question'])
            lsti.append(int(json_str['image_id']))
    with open(path2a) as fread:
        for line in fread:
            json_str = json.loads(line)
            lsta.append(json_str['answer'])

    return lsti, lstq, lsta


def main():
    # TODO: Parsing the list arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-qadir', required=True, type=str)
    parser.add_argument('-imgdir', required=True, type=str)
    parser.add_argument('-outdir', required=True, type=str)
    parser.add_argument('-dataname', required=True, type=str)
    parser.add_argument('-preprocess', default=True, type=bool)
    parser.add_argument('-buildimg', default=True, type=bool)
    parser.add_argument('-imgsparse', default=True, type=bool)
    parser.add_argument('-h5key', required=True, type=str)
    parser.add_argument('-maxlenquest', required=True, type=int)
    args = parser.parse_args()

    path2qadir = args.qadir
    path2imgdir = args.imgdir
    path2outputdir = args.outdir

    dataname = args.dataname
    maxlen = args.maxlenquest

    if not os.path.exists(path2outputdir):
        os.mkdir(path2outputdir)

    # 1/ Load datasets
    lst_img_train, lst_quest_train, lst_ans_train = load_data(path2qadir, dataname, datapart='train')
    lst_img_val, lst_quest_val, lst_ans_val = load_data(path2qadir, dataname, datapart='val')
    lst_img_test, lst_quest_test, lst_ans_test = load_data(path2qadir, dataname, datapart='test')

    # 2/ Preprocess
    if args.preprocess:
        lst_quest_train = preprocess(lst_quest_train, is_lmz=False)
        lst_ans_train = preprocess(lst_ans_train, is_lmz=False)
        lst_quest_val = preprocess(lst_quest_val, is_lmz=False)
        lst_ans_val = preprocess(lst_ans_val, is_lmz=False)
        lst_quest_test = preprocess(lst_quest_test, is_lmz=False)
        lst_ans_test = preprocess(lst_ans_test, is_lmz=False)
    # 3/ Convert questions and answers to numerical format
    q_train, a_train, qword_dict, aword_dict = conv_txtdata2num(lst_quest_train, lst_ans_train, maxlen)
    q_val, a_val = conv_txtdata2num_withdict(lst_quest_val, lst_ans_val, maxlen, qword_dict, aword_dict)
    q_test, a_test = conv_txtdata2num_withdict(lst_quest_test, lst_ans_test, maxlen, qword_dict, aword_dict)

    # 4/ Prepare Image Data
    img_feat_train, img_idx_train = load_imgfeat(path2imgdir, datapart='train')
    #   4.1/ Calculating mean and std of images feature in training dataset
    img_mean_train = np.mean(img_feat_train, axis=0)
    img_std_train = np.std(img_feat_train, axis=0)
    for i in range(img_std_train.shape[0]):
        if img_std_train[i] == 0.0:
            img_std_train[i] = 1.0

    img_feat_test, img_idx_test = load_imgfeat(path2imgdir, datapart='val')

    img_feat = np.concatenate((img_feat_train, img_feat_test), axis=0)
    img_idx = np.concatenate((img_idx_train, img_idx_test), axis=0)

    #   4.2/ Save to h5 file
    h5key = args.h5key
    fout = h5py.File(os.path.join(path2outputdir, 'image-features-all.h5'), 'w')
    if args.imgsparse:
        img_feat_sparse = sparse.csr_matrix(img_feat)
        fout[h5key + '_shape'] = img_feat_sparse._shape
        fout[h5key + '_data'] = img_feat_sparse.data
        fout[h5key + '_indices'] = img_feat_sparse.indices
        fout[h5key + '_indptr'] = img_feat_sparse.indptr
        fout[h5key + '_mean'] = img_mean_train.astype('float32')
        fout[h5key + '_std'] = img_std_train.astype('float32')
    else:
        fout[h5key + '_data'] = img_feat
        fout[h5key + '_mean'] = img_mean_train.astype('float32')
        fout[h5key + '_std'] = img_std_train.astype('float32')

    # 4.3/ Mapping
    i_train = map_imgid2idx(lst_img_train, img_idx)
    i_val = map_imgid2idx(lst_img_val, img_idx)
    i_test = map_imgid2idx(lst_img_test, img_idx)

    # 5/ Combine Images, Questions, Answers in numerical format into one numpy file
    #    5.1/ Training Data
    input_train = np.concatenate((i_train, q_train), axis=1)
    target_train = a_train
    np.save(os.path.join(path2outputdir, 'train.npy'), np.array((input_train, target_train), dtype=object))
    #    5.2/ Validation Data
    input_val = np.concatenate((i_val, q_val), axis=1)
    target_val = a_val
    np.save(os.path.join(path2outputdir, 'dev.npy'), np.array((input_val, target_val), dtype=object))
    #    5.3/ Test Data
    input_test = np.concatenate((i_test, q_test), axis=1)
    target_test = a_test
    np.save(os.path.join(path2outputdir, 'val.npy'), np.array((input_test, target_test), dtype=object))
    #    5.4/ Dictionary
    with open(os.path.join(path2outputdir, 'qdict.pkl'), 'w') as fwrite:
        pickle.dump(qword_dict, fwrite)
    with open(os.path.join(path2outputdir, 'adict.pkl'), 'w') as fwrite:
        pickle.dump(aword_dict, fwrite)


if __name__ == '__main__':
    main()
