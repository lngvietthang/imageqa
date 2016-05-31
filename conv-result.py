import argparse
import pickle
import numpy as np
import json
import operator
import h5py


def main():
    # TODO: Parsing the list arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ansval', required=True, type=str)
    parser.add_argument('-infile', required=True, type=str)
    parser.add_argument('-outfile', required=True, type=str)
    parser.add_argument('-ansdict', required=True, type=str)
    args = parser.parse_args()

    path2ansval = args.ansval
    path2infile = args.infile
    path2outfile = args.outfile
    path2ansdict = args.ansdict

    # Load ans dict
    with open(path2ansdict) as fread:
        ansdict = pickle.load(fread)

    ansdict.pop('padding', None)
    ansdict_sorted = sorted(ansdict.items(), key=operator.itemgetter(1)) # Sorted by value
    columns = []
    for k, v in ansdict_sorted:
        columns.append(k)

    # Load results
    data = np.load(path2infile)

    # Load list of answers in val set to get question IDs
    lst_quest_ids = []
    with open(path2ansval) as fread:
        for line in fread:
            json_str = json.loads(line)
            lst_quest_ids.append(json_str['question_id'])

    fOut = h5py.File(path2outfile, 'w')
    fOut['columns'] = [ans.encode('utf8') for ans in columns]
    fOut['data'] = data
    fOut['index'] = lst_quest_ids
    fOut.flush()
    fOut.close()

    print 'DONE!'


if __name__ == '__main__':
    main()