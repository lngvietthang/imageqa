import json
import h5py
import os
import argparse

import numpy as np
from scipy import sparse

from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer

verbose = os.environ.get('VERBOSE', 'no') == 'yes'
debug = os.environ.get('DEBUG', 'no') == 'yes'

def preprocess(lines, isLmz = True):
    lines = [wordpunct_tokenize(line) for line in lines]
    if isLmz:
        lemmatizer = WordNetLemmatizer()
        lines = [[lemmatizer.lemmatize(word) for word in line] for line in lines]

    lines = [' '.join([word for word in line]) for line in lines]

    return lines

def convTxt2Num(lstTxt): # <- The problem <<
    tkz = Tokenizer()
    tkz.fit_on_texts(lstTxt)
    lstNum = tkz.texts_to_sequences(lstTxt)

    return (lstNum, tkz)

def convDataTxt2Num(lstQ, lstA, maxLenQ):
    lstQ, tkzQ = convTxt2Num(lstQ)
    dictQ = tkzQ.word_index
    dictQ['UNK'] = max(dictQ.values()) + 1
    vocabSize = len(dictQ)
    lstA, tkzA = convTxt2Num(lstA)
    dictA = tkzA.word_index
    dictA['UNK'] = max(dictA.values()) + 1
    nbofAns = len(dictA)
    lstA = [[idx - 1 for idx in ans] for ans in lstA]

    arrQ = np.zeros((len(lstQ), maxLenQ), dtype=int)
    for i in range(len(lstQ)):
        for j in range(maxLenQ):
            arrQ[i, j] = lstQ[i][j]

    arrA = np.array(lstA).reshape(len(lstA), 1)

    result = {}
    result['arrQ'] = arrQ
    result['arrA'] = arrA
    result['dictQ'] = dictQ
    result['dictA'] = dictA

    # Verbose
    if verbose:
        print 'Question Vocabulary Size:', vocabSize
        print 'Number of candidate answers:', nbofAns

    return result

def loadData(path2DataDir, dataName, datapart):
    path2Q = os.path.join(path2DataDir, dataName + '_' + datapart + '_questions.json')
    path2A = os.path.join(path2DataDir, dataName + '_' + datapart + '_answers.json')

    # Load Training Questions and Answers
    lstQ = []
    lstA = []
    lstI = []
    with open(path2Q) as fread:
        for line in fread:
            jsonStr = json.loads(line)
            lstQ.append(jsonStr['questions'])
            lstI.append(int(jsonStr['image_id']))
    with open(path2A) as fread:
        for line in fread:
            jsonStr = json.loads(line)
            lstQ.append(jsonStr['answer'])

    return (lstI, lstQ, lstA)


def main():
    # TODO: Parsing the list arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-qadir', required=True, type=str)
    parser.add_argument('-imgdir', required=True, type=str)
    parser.add_argument('-outdir', required=True, type=str)
    parser.add_argument('-dataname', required=True, type=str)
    parser.add_argument('-preprocess', default=True, type=bool)
    parser.add_argument('-imgSparse', default=True, type=bool)
    parser.add_argument('-h5key', required=True, type=str)
    parser.add_argument('-maxLenQuest', required=True, type=int)
    args = parser.parse_args()

    path2QADir = args.qadir
    path2ImgDir = args.imgdir
    path2OutputDir = args.outdir

    dataName = args.dataname

    # 1/ Load datasets
    lstITrain, lstQTrain, lstATrain = loadData(path2QADir, dataName, datapart='train')
    lstIVal, lstQVal, lstAVal = loadData(path2QADir, dataName, datapart='val')
    lstITest, lstQTest, lstATest = loadData(path2QADir, dataName, datapart='test')

    # 2/ Preprocess
    if args.preprocess:
        lstQTrain = preprocess(lstQTrain)
        lstATrain = preprocess(lstATrain, isLmz = False)
        lstQVal = preprocess(lstQVal)
        lstAVal = preprocess(lstAVal, isLmz = False)
        lstQTest = preprocess(lstQTest)
        lstATest = preprocess(lstATest, isLmz = False)
    # 3/ Convert data to numerical format
    maxLenQ = args.maxLenQuest
    dataQATrain = convDataTxt2Num(lstQTrain, lstATrain, maxLenQ)

if __name__ == '__main__':
    main()
