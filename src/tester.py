import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys, time
import numpy as np
from scipy import stats
import pandas as pd


from classifier import Classifier


def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)


def load_dataset(filename):
    """ Download the date: list of texts with scores."""
    headers = ['polarity', 'text']
    examples = pd.read_csv(filename, encoding="utf-8", sep='\t', names=headers)
    # print distributions by rating or class
    # print(sentences.groupby('polarity').nunique())
    # return the list of rows : row = label and text
    return examples.text.to_list(), examples.polarity.to_list()


def eval_list(glabels, slabels):
    if (len(glabels) != len(slabels)):
        print("\nWARNING: label count in system output (%d) is different from gold label count (%d)\n" % (
        len(slabels), len(glabels)))
    n = min(len(slabels), len(glabels))
    incorrect_count = 0
    for i in range(0, n):
        if slabels[i] != glabels[i]: incorrect_count += 1
    acc = (n - incorrect_count) / n
    acc = acc * 100
    return acc


def train_and_eval_dev_test(trainfile, devfile, testfile, run_id):
    train_texts, train_labels = load_dataset(trainfile)
    dev_texts, dev_labels = load_dataset(devfile)
    classifier = Classifier()
    print("\n")
    # Training
    print("RUN: %s" % str(run_id))
    print(f"  {run_id}.1. Training the classifier (train data size = {len(train_texts)})...")
    classifier.train(train_texts, train_labels, dev_texts, dev_labels)
    print()
    print(f"  {run_id}.2. Evaluation on the dev dataset (dev data size = {len(dev_texts)})..." )
    predicted_labels = classifier.predict(dev_texts)
    devacc = eval_list(dev_labels, predicted_labels)
    print("       Acc.: %.2f" % devacc)
    testacc = -1
    if testfile is not None:
        # Evaluation on the test data
        test_texts, test_labels = load_dataset(testfile)
        print(f"  {run_id}.3. Evaluation on the test dataset (test data size = {len(test_texts)})...")
        predicted_labels = classifier.predict(test_texts)
        testacc = eval_list(test_labels, predicted_labels)
        print("       Acc.: %.2f" % testacc)
    print()
    classifier = None
    return (devacc, testacc)

if __name__ == "__main__":
    set_reproducible()
    datadir = "../data/"
    trainfile =  datadir + "frdataset1_train.csv"
    devfile =  datadir + "frdataset1_dev.csv"
    # testfile =  datadir + "frdataset1_test.csv"
    testfile = None
    # Basic checking
    start_time = time.perf_counter()
    n = 5
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    devaccs = []
    testaccs = []
    for i in range(n):
        res = train_and_eval_dev_test(trainfile, devfile, testfile, i+1)
        devaccs.append(res[0])
        testaccs.append(res[1])
    print('\nCompleted %d runs.' % n)
    print("Dev accs:", devaccs)
    print("Test accs:", testaccs)
    print()
    print("Mean Dev Acc.: %.2f (%.2f)\tMean Test Acc.: %.2f (%.2f)" % (np.mean(devaccs), np.std(devaccs), np.mean(testaccs), np.std(testaccs)))
    total_exec_time = (time.perf_counter()-start_time)
    print("\nExec time: %.2f s. ( %d per run )" % (total_exec_time, total_exec_time/n))



