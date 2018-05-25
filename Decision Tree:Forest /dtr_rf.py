#!/usr/bin/python

import os
import argparse
import sys
import random
import cv2
import glob
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

#################################################
# module: dtr_rf.py
# Cole Shepherd
# A01895001
# The DTR Validation yielded results in the 80% range. The Random Forests yielded in the HIGh 90%.
# I would say in this case the more accurate of the two are the Random Forests.
##################################################

## ================== LOADING DATA =======================

## Change the value of BEE_DIR accordingly
BEE_DIR = './BEE_DIR'
BEE = []
DATA = []
TARGET = []


def load_data(imgdir):

    global DATA, TARGET

    list = []

    sub_dir = ['no_bee', 'yes_bee']
    for label, class_names in enumerate(sub_dir, start=0):
        vector_dir = os.path.join(imgdir, class_names, '*.png')
        all_files = glob.glob(vector_dir)
        for f in all_files:
            image = cv2.imread(f, 0).flatten()
            normalized_image = (image / float(255))
            list.append((np.ndarray.tolist(normalized_image), class_names))

    random.shuffle(list)

    for item in list:
        if item[1] == 'no_bee':
            DATA.append(item[0])
            TARGET.append(0)
        elif item[1] == 'yes_bee':
            DATA.append(item[0])
            TARGET.append(1)

    DATA = np.asarray(DATA)
    TARGET = np.asarray(TARGET)

## ===================== DECISION TREES ==============================

def run_dtr_train_test_split(data, target, n, test_size):
    returnList = []
    for i in xrange(n):
        train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size, random_state=random.randint(0, 1000))

        clf = tree.DecisionTreeClassifier(random_state=random.randint(0, 1000))
        dtr = clf.fit(data, target)

        dt = dtr.fit(train_data, train_target)

        acc_preds = sum(dt.predict(test_data) == test_target)
        print('train/test run %d: accuracy = %f' % (i, acc_preds / float(len(test_target))))
        returnList.append((i, acc_preds / float(len(test_target))))
        print('------------------------------------------')
    return returnList

def plot_dtr_train_test_split(acc_pred_list, n, test_size):

    experimentNumber = [acc_pred_list[x][0] for x in xrange(len(acc_pred_list))]
    accuracy = [acc_pred_list[x][1] for x in xrange(len(acc_pred_list))]

    fig1 = plt.figure(1)
    fig1.suptitle('DTR Train/Test Split; Test Size = %f' % test_size)
    axes = plt.axes()
    axes.set_ylabel("accuracy")
    axes.set_xlabel("experiment number")
    axes.set_xlim([0, n-1])
    axes.set_ylim([-.1, 1.1])
    axes.xaxis.set_ticks(np.arange(0, n-1, 1))
    axes.yaxis.set_ticks(np.arange(-.1, 1.1, .1))

    plt.grid()
    plt.scatter(experimentNumber, accuracy)

    plt.show()


def run_dtr_cross_validation(data, target, test_size):
    returnList = []
    clf = tree.DecisionTreeClassifier(random_state=random.randint(0, 1000))
    dtr = clf.fit(data, target)

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size,
                                                                        random_state=random.randint(0, 1000))

    dt = dtr.fit(train_data, train_target)

    print('cross-validation run:')

    for cv in xrange(5, 21):
            cross_val = cross_val_predict(dt, test_data, test_target, cv=cv)
            acc = sum(cross_val == test_target) / float(len(test_target))
            print('num_folders %d, accuracy = %f' % (cv, acc))
            print('------------------------------------------')
            returnList.append((cv, acc))

    return returnList



def plot_dtr_cross_validation(nf_acc_list, nf_lower, nf_upper, test_size):

    folds = [nf_acc_list[x][0] for x in xrange(len(nf_acc_list))]
    accuracy = [nf_acc_list[x][1] for x in xrange(len(nf_acc_list))]

    fig2 = plt.figure(2)
    fig2.suptitle('DTR Cross Validation; Test Size = %f' % test_size)
    axes = plt.axes()
    axes.set_xlabel("folds")
    axes.set_ylabel("accuracy")
    axes.set_xlim([nf_lower,nf_upper])
    axes.set_ylim([-.1, 1.1])
    axes.xaxis.set_ticks(np.arange(nf_lower, nf_upper, 1))
    axes.yaxis.set_ticks(np.arange(-.1, 1.1, .1))
    plt.grid()
    plt.scatter(folds, accuracy)

    plt.show()

def compute_cr_cm(data, target, test_size):
    clf = tree.DecisionTreeClassifier(random_state=0)
    dtr = clf.fit(data, target)

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size,
                                                                        random_state=random.randint(0, 1000))

    dt = dtr.fit(train_data, train_target)
    clf_expected = test_target
    clf_predicted = dt.predict(test_data)
    cm = confusion_matrix(clf_expected, clf_predicted)

    print('Classification report for decision tree %s:\n%s\n' % (
    dt, classification_report(clf_expected, clf_predicted)))
    print('Confusion matrix:\n%s' % cm)
    print('\n')


## ================= RANDOM FORESTS ==============================

def create_rfs(n, num_trees, data, target):
    random_forests = []
    test_sizes = (0.20, 0.25, 0.30, 0.35, 0.40)
    for i in xrange(n):
        tsize = random.choice(test_sizes)
        train_data, _, train_target, _ = \
            train_test_split(data, target, test_size=tsize,
                             random_state=random.randint(0, 1000))
        clf = RandomForestClassifier(n_estimators=num_trees,
                                     random_state=random.randint(0, 1000))
        random_forests.append(clf.fit(train_data, train_target))
    return random_forests

def classify_with_rfs_aux(rand_forests, data_item):
    clfs = [dt.predict(data_item.reshape(1, -1))[0] for dt in rand_forests]
    return clfs

def classify_with_rfs(rfs, data, target):
    i = random.randint(0, DATA.shape[0] - 1)
    rf_classifications = classify_with_rfs_aux(rfs, DATA[i])
    return rf_classifications, target[i]

def majority_vote(rf_classifications):
    counts = {0:0, 1:0}
    for rfc in rf_classifications:
        counts[rfc] += 1
    return sorted([kv for kv in counts.iteritems()],
                key=lambda x: x[1],
                reverse=True)[0][0]

def run_rf_mv_experiments(rfs, data, target, n):
    acc = 0
    for _ in xrange(n):
        rfc, t = classify_with_rfs(rfs, data, target)
        v = majority_vote(rfc)
        if v == t:
            acc += 1
    return acc / float(n)

def collect_rf_mv_stats(rfs_list, data, target, n):
    num_trees_acc_list = []
    for num_trees, rfs in rfs_list:
        num_trees_acc_list.append((num_trees,run_rf_mv_experiments(rfs,data,target,n)))
    return num_trees_acc_list


def create_rf_list(ntrees_in_rf, data, target):
    list = []
    for x in [5, 10, 15, 20, 25, 30, 35, 40]:
        list.append((x, create_rfs(ntrees_in_rf, x, data, target)))
    return list


def plot_rf_mv_stats(rf_mv_stats, num_trees_lower, num_trees_upper):
    folds = [rf_mv_stats[x][0] for x in xrange(len(rf_mv_stats))]
    accuracy = [rf_mv_stats[x][1] for x in xrange(len(rf_mv_stats))]

    fig2 = plt.figure(2)
    fig2.suptitle('Random Forests Majority Vote Stats')
    axes = plt.axes()
    axes.set_xlabel("num trees in random forest")
    axes.set_ylabel("accuracy")
    axes.set_xlim([num_trees_lower,num_trees_upper])
    axes.set_ylim([-.1, 1.1])
    axes.xaxis.set_ticks(np.arange(num_trees_lower, num_trees_upper, 1))
    axes.yaxis.set_ticks(np.arange(-.1, 1.1, .1))
    plt.grid()
    plt.scatter(folds, accuracy)

    plt.show()
    pass

#
# if __name__ == "__main__":
#     #
#     # nf_acc_list = run_dtr_cross_validation(DATA, TARGET, 0.3)
#     # plot_dtr_cross_validation(nf_acc_list, 5, 20, 0.3)
#
#     # exp_acc_list = run_dtr_train_test_split(DATA, TARGET, 10, 0.3)
#     # plot_dtr_train_test_split(exp_acc_list, 10, 0.3)
#     pass
