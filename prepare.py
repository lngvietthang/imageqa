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
  '''

  '''
  lines = [wordpunct_tokenize(line) for line in lines]
  if isLmz:
    lemmatizer = WordNetLemmatizer()
    lines = [[lemmatizer.lemmatize(word) for word in line] for line in lines]

  lines = [' '.join([word for word in line]) for line in lines]

  return lines

def loadData(config, path2DataDir, dataName, datapart):
  '''Loading dataset
  Parameters:
  -----------
  '''
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

def convTxt2Num(lstTxt):
  '''
  '''
  tkz = Tokenizer()
  tkz.fit_on_texts(lstTxt)
  lstNum = tkz.texts_to_sequences(lstTxt)

  return (lstNum, tkz)

def convTxt2NumWithDict(lstTxt, _dict):
  '''
  '''
  #lstNum = tkz.texts_to_sequences(lstTxt)

  lstTxt = [line.split() for line in lstTxt]
  results = []
  for line in lstTxt:
    nline = []
    for word in line:
      if _dict.has_key(word):
        nline.append(_dict[word])
      else:
        nline.append(_dict['UNK'])
    results.append(nline)

  return results

def convDataTxt2Num(lstQ, lstA, maxLenQ):
  '''
  '''
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

def convDataTxt2NumWithDict(lstQ, lstA, maxLenQ, dictQ, dictA):
  '''
  '''
  #
  lstQ = convTxt2NumWithDict(lstQ, dictQ)
  lstA = convTxt2NumWithDict(lstA, dictA)
  lstA = [[idx - 1 for idx in ans] for ans in lstA]

  arrQ = np.zeros((len(lstQ), maxLenQ), dtype=int)
  for i in range(len(lstQ)):
    for j in range(maxLenQ):
      arrQ[i, j] = lstQ[i][j]

  arrA = np.array(lstA).reshape(len(lstA), 1)

  result = {}
  result['arrQ'] = arrQ
  result['arrA'] = arrA

  return result

def loadImgFeat(path2DataDir, datapart):
  '''
  '''
  path2ImgFeat = os.path.join(path2DataDir, 'mscoco2014_' + datapart + '_vgg-fc7.h5')
  fData = h5py.File(path2ImgFeat)
  data = fData['data'][:]
  index = fData['index'][:]

  return (data, index)

def mapImgID2Idx(lstI, imgIdx):
  '''
  '''

  result = np.zeros((len(lstI), 1), dtype=int)

  for i, ins in enumerate(lstI):
    idx = np.where(imgIdx == ins)
    if len(idx) == 1:
      result[i] = idx[0]
    else:
      print "[1] Problem in the list of image ids."
      raise ValueError('[1] Problem in the list of image ids')

  return result

def main():
  #TODO:
  # 1/ Parsing the list arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-qadir', required = True, type = str)
  parser.add_argument('-imgdir', required = True, type = str)
  parser.add_argument('-outdir', required = True, type = str)
  parser.add_argument('-dataname', required = True, type = str)
  parser.add_argument('-preprocess', default=True, type=bool)
  parser.add_argument('-imgSparse', default=True, type=bool)
  parser.add_argument('-h5key', required = True, type=str)
  parser.add_argument('-maxLenQuest', required=True, type=int)
  args = parser.parse_args()

  path2QADir = args.qadir
  path2ImgDir = args.imgdir
  path2OutputDir = args.outdir

  dataName = args.dataname

  # 2/ Load datasets
  lstITrain, lstQTrain, lstATrain = loadData(path2QADir, \
                                             dataName, \
                                             datapart = 'train'\
                                             )
  lstIVal, lstQVal, lstAVal = loadData(path2QADir, \
                                       dataName, \
                                       datapart = 'val' \
                                       )
  lstITest, lstQTest, lstATest = loadData(path2QADir, \
                                          dataName, \
                                          datapart = 'test' \
                                          )
  #   2.1/ Preprocess
  if args.preprocess:
    lstQTrain = preprocess(lstQTrain)
    lstATrain = preprocess(lstATrain, isLmz = False)
    lstQVal = preprocess(lstQVal)
    lstAVal = preprocess(lstAVal, isLmz = False)
    lstQTest = preprocess(lstQTest)
    lstATest = preprocess(lstATest, isLmz = False)

  # Combine Training and Validation dataset
  #lstI = lstITrain + lstIVal
  #lstQ = lstQTrain + lstQVal
  #lstA = lstATrain + lstAVal

  # 3/ Convert data to numerical format
  maxLenQ = args.maxLenQuest
  dataQATrain = convDataTxt2Num(lstQTrain, \
                                lstATrain, \
                                maxLenQ \
                                )
  #dataQA = convDataTxt2Num(lstQ, \
  #                         lstA, \
  #                         maxLenQ \
  #                         )

  dataQAVal = convDataTxt2NumWithDict(lstQVal, \
                                      lstAVal, \
                                      maxLenQ, \
                                      dictQ = dataQATrain['dictQ'], \
                                      dictA = dataQATrain['dictA'] \
                                      )

  dataQATest = convDataTxt2NumWithDict(lstQTest, \
                                       lstATest, \
                                       maxLenQ, \
                                       dictQ = dataQATrain['dictQ'], \
                                       dictA = dataQATrain['dictA'] \
                                       )

  #dataQATest2 = convDataTxt2NumWithDict(lstQTest, \
  #                                      lstATest, \
  #                                      maxLenQ, \
  #                                      dictQ = dataQA['dictQ'], \
  #                                      dictA = dataQA['dictA'] \
  #                                      )


  # 4/ Prepare Image Data
  imgFeatTrain, imgIdxTrain = loadImgFeat(path2ImgDir, \
                                          datapart = 'train' \
                                          )
  #   4.1/ Calculating mean and std of images feature in training dataset
  imgMeanTrain = np.mean(imgFeatTrain, axis = 0)
  imgStdTrain = np.std(imgFeatTrain, axis = 0)
  for i in range(imgStdTrain.shape[0]):
    if imgStdTrain[i] == 0.0:
      imgStdTrain[i] = 1.0

  imgFeatTest, imgIdxTest = loadImgFeat(path2ImgDir, \
                                        datapart = 'val' \
                                        )

  imgFeat = np.concatenate((imgFeatTrain, imgFeatTest), axis = 0)
  imgIdx = np.concatenate((imgIdxTrain, imgIdxTest), axis = 0)

  #   4.2/ Save to h5 file
  h5key = args.h5key
  fOut = h5py.File(os.path.join(path2OutputDir, \
                                'image-features-all.h5'), 'w')
  if args.imgsparse:
    imgFeatSparse = sparse.csr_matrix(imgFeat)
    fOut[h5key + '_shape'] = imgFeatSparse._shape
    fOut[h5key + '_data'] = imgFeatSparse.data
    fOut[h5key + '_indices'] = imgFeatSparse.indices
    fOut[h5key + '_indptr'] = imgFeatSparse.indptr
    fOut[h5key + '_mean'] = imgMeanTrain.astype('float32')
    fOut[h5key + '_std'] = imgStdTrain.astype('float32')
  else:
    fOut[h5key + '_data'] = imgFeat
    fOut[h5key + '_mean'] = imgMeanTrain.astype('float32')
    fOut[h5key + '_std'] = imgStdTrain.astype('float32')

  #   4.3/ Mapping
  arrITrain = mapImgID2Idx(lstITrain, imgIdx)
  arrIVal = mapImgID2Idx(lstIVal, imgIdx)
  arrITest = mapImgID2Idx(lstITest, imgIdx)
  arrI = mapImgID2Idx(lstI, imgIdx)

  # 5/ Combine Images, Questions, Answers in numerical format into one numpy file
  #    5.1/ Training Data
  inputTrain = np.concatenate((arrITrain, dataQATrain['arrQ']), axis=1)
  targetTrain = dataQATrain['arrA']
  np.save(os.path.join(path2OutputDir, 'train.npy'), \
          np.array((inputTrain, targetTrain), dtype=object) \
          )
  #    5.2/ Validation Data
  inputVal = np.concatenate((arrIVal, dataQAVal['arrQ']), axis=1)
  targetVal = dataQAVal['arrA']
  np.save(os.path.join(path2OutputDir, 'valid.npy'), \
          np.array((inputVal, targetVal), dtype=object) \
          )
  #    5.3/ Test Data
  inputTest = np.concatenate((arrITest, dataQATest['arrQ']), axis=1)
  targetTest = dataQATest['arrA']
  np.save(os.path.join(path2OutputDir, 'test.npy'), \
          np.array((inputTest, targetTest), dtype=object) \
          )
  #    5.4/ Train + Validation  Data
  inputVT = np.concatenate((arrI, dataQA['arrQ']), axis=1)
  targetVT = dataQA['arrA']
  np.save(os.path.join(path2OutputDir, 'train-all.npy'), \
          np.array((inputVT, targetVT), dtype=object) \
          )
  #    5.5/ Test Data with Train+Validation Dict
  inputTest2 = np.concatenate((arrITest, dataQATest2['arrQ']), axis=1)
  targetTest2 = dataQATest2['arrA']
  np.save(os.path.join(path2OutputDir, 'test-all.npy'), \
          np.array((inputTest2, targetTest2), dtype=object) \
          )

if __name__ == '__main__':
  main()
