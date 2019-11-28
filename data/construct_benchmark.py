# -*- coding: UTF-8 -*-
"""

Construct benchmark for transfer learning for anomaly detection.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import pandas as pd
import sklearn as sk
import math, os, sys
import random
import itertools
import operator
from collections import Counter

from tqdm import tqdm

from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


# -------------
# VARIABLES
# -------------

INPUT_DIR = '/Users/vincent/Benchmark_data/outlier_detection/raw_datasets/'
OUTPUT_DIR = '/Users/vincent/Benchmark_data/outlier_detection/set7/'

MIN_SIZE = 500
MIN_SUBSET = 50
A_PERCENT = 0.1

# dictionary with for each dataset: (N1, A1) method 5
CLASS_DICT_M5 = {
    'abalone': (2, 1),
    'covertype': (1, 0),
    'gas_sensors': (4, 5),
    'gesture_segmentation': (4, 2),
    'hand_posture': (4, 1),
    'landsat_satellite': (5, 3),
    'handwritten_digits': (1, 4),
    'letter_recognition': (1, 17),
    'pen_digits': (2, 1),
    'satimage': (5, 3),
    'segment': (2, 4),
    'sense_IT_acoustic': (2, 0),
    'sense_IT_seismic': (2, 1),
    'sensorless': (5, 3),
    'shuttle': (0, 2),
    'waveform': (2, 0),
    'poker': (0, 2) # Takes long computation time!
    }


# -------------
# MAIN
# -------------

def main():
    # read each .csv file in the input directory
    files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
    csv_files = [f for f in files if '.csv' in f]

    # for each file construct the target sets
    for f in csv_files:
        print('')
        print('Processing file:', f)
        name = f.split('.')[0]

        # read the file and gather features, labels
        if name == 'poker':
            features, scaled_features, labels, ncl = load_data(os.path.join(INPUT_DIR, f), drop_classes=[7, 8, 9])
            # classes 0, 1 needs subsampling
            features, scaled_features, labels = subsample_data(features, scaled_features, labels, {0: 0.1, 1: 0.1})
        else:
            features, scaled_features, labels, ncl = load_data(os.path.join(INPUT_DIR, f))
        n_samples, n_dim = features.shape
        print('Number of datapoints:', n_samples)
        print('Number of features:', n_dim)
        print('Number of classes:', ncl)
        print('Datapoints per class:', Counter(labels))

        # get class combination from class dictionary
        if name in CLASS_DICT_M5.keys():
            previously_picked_classes = CLASS_DICT_M5[name]
        else:
            previously_picked_classes = None

        # construct target sets: select the normal and anomaly class - normals and anomalies contain indices that span entire dataset!
        normals_ixs, anomalies_ixs, picked_classes = construct_target_sets_method_5(scaled_features, labels, ncl, n_samples, n_dim, previously_picked_classes)

        # pick source and target classes
        possible_classes = set([i for i in range(ncl)]) - set(picked_classes)
        n1, a1 = picked_classes[0], picked_classes[1]
        if ncl == 3:
            assert len(possible_classes) == 1, 'Error - problem with amount of classes'
            # pick the one remaining class as n2
            n2 = random.sample(possible_classes, 1)[0]
            # class combos to sample datasets
            combos = [(n1, a1), (n2, a1), (n2, n1), ((n1, n2), a1)]  # 4 possible source domains
            combo_names = ['n1_a1', 'n2_a1', 'n2_n1', 'n12_a1']
        else:
            # select the largest of the remaining classes as n2
            class_sizes = {k: v for k, v in Counter(labels).items() if k in possible_classes}
            n2 = max(class_sizes.items(), key=operator.itemgetter(1))[0]
            # select the remaining class or the largest of the remaining classes as a2
            possible_classes = possible_classes - set([n2])
            class_sizes = {k: v for k, v in Counter(labels).items() if k in possible_classes}
            a2 = max(class_sizes.items(), key=operator.itemgetter(1))[0]
            # class combos to sample datasets
            combos = [(n1, a1), (n1, a2), (n2, a1), (n2, a2), (n2, n1), ((n1, n2), (a1, a2))]  # 6 possible source domains
            combo_names = ['n1_a1', 'n1_a2', 'n2_a1', 'n2_a2', 'n2_n1', 'n12_a12']

        # path to source and target set
        if not os.path.exists(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100)))):
            os.makedirs(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))))
        if not os.path.exists(os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'target')):
             os.makedirs(os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'target'))
        if not os.path.exists(os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'source')):
             os.makedirs(os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'source'))

        # define sampling n1 and a1 for source and target
        # sampling without replacement! Source and target instances cannot be equal!
        tgt_n1, src_n1 = {}, {}
        tgt_a1, src_a1 = {}, {}
        for i in [0, 1]:
            # normals
            ixs = normals_ixs[i].copy()
            random.shuffle(ixs)
            ln = len(ixs)
            tgt_n1[i] = ixs[:int(ln/2)]     # half is target, half is source (no overlap)
            src_n1[i] = ixs[int(ln/2):]

            # anomalies
            ixs = anomalies_ixs[i].copy()
            random.shuffle(ixs)
            la = len(ixs)
            tgt_a1[i] = ixs[:int(la/2)]     # half is target, half is source (no overlap)
            src_a1[i] = ixs[int(la/2):]

        # construct the target set(s)
        construct_data_sets(tgt_n1, tgt_a1, features, A_PERCENT, os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'target'), name, 10)

        # construct the source set(s) - construct multiple times to avoid lucky sampling
        for i in range(5):
            construct_source_sets(src_n1, src_a1, picked_classes, combos, combo_names, features, labels, A_PERCENT, os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'source'), name, i)


# -------------
# FUNCTIONS
# -------------

def construct_target_sets_method_4(features, labels, ncl, n_samples, n_dim, final_features, dataset_name=None):
    """ Construct the target sets: most confusion between classes. """

    class_combinations = np.unique(labels)

    normals = None
    anomalies = None

    # investigate each combination of classes
    confusion = np.inf
    miss_classified = 0
    best_combination = None
    for each in itertools.combinations(class_combinations, 2):
        # use precalculated or determined classes
        skip_rest = False
        if dataset_name in CLASS_DICT.keys():
            each = CLASS_DICT[dataset_name]
            skip_rest = True

        print('iteration with:', each)

        # select the classes
        idx_c1 = np.where(labels == each[0])[0]
        idx_c2 = np.where(labels == each[1])[0]
        idx = np.concatenate((idx_c1, idx_c2))
        selected_features = features[idx, :]
        selected_labels = labels[idx]

        # final features
        ff_class1 = final_features[idx_c1, :]
        ff_class2 = final_features[idx_c2, :]

        # cross-validation to separate classes
        class_probabilities = np.zeros(n_samples)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_idx, test_idx in skf.split(selected_features, selected_labels):
            # fit + predict
            clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)
            clf.fit(selected_features[train_idx, :], selected_labels[train_idx])
            y_prob = clf.predict_proba(selected_features[test_idx, :])
            c = np.where(clf.classes_ == each[0])[0][0]
            class_probabilities[test_idx] = y_prob[:, c]

        # calculate confusion class 1
        idx1 = np.where(selected_labels == each[0])[0]
        cp = class_probabilities[idx1]
        idx_cc_class1 = np.where(cp >= 0.5)[0]
        new_miss_classified = len(idx1) - len(idx_cc_class1)
        new_confusion = np.sum(cp[idx_cc_class1])

        # calculate confusion class 2
        idx = np.where(selected_labels == each[1])[0]
        cp = 1.0 - class_probabilities[idx]
        idx_cc_class2 = np.where(cp > 0.5)[0]
        new_miss_classified += len(idx) - len(idx_cc_class2)
        new_confusion += np.sum(cp[idx_cc_class2])

        assert len(np.intersect1d(idx1, idx)) == 0, 'Error - normals and anomalies not off different class!'

        # adapt confusion etc.
        if new_confusion < confusion:
            # update the confusion metric
            confusion = new_confusion
            miss_classified = new_miss_classified
            best_combination = each

            # update the set of normals and anomalies
            # smallest class is the anomaly class
            if len(idx_cc_class1) > len(idx_cc_class2):
                normals = ff_class1[idx_cc_class1, :]
                anomalies = ff_class2[idx_cc_class2, :]
            else:
                normals = ff_class2[idx_cc_class2, :]
                anomalies = ff_class1[idx_cc_class1, :]

        if skip_rest:
            break

    # print classes picked
    print('Classes picked:', best_combination)

    return {0: normals, 1: np.array([])}, {0: anomalies, 1: np.array([])}


def construct_target_sets_method_5(features, labels, ncl, n_samples, n_dim, picked_classes=None):
    """ Construct the target sets: most confusion between classes. """

    class_combinations = np.unique(labels)

    normals = None
    anomalies = None

    # investigate each combination of classes
    skip_rest = False
    confusion = 0
    best_combination = None
    for each in itertools.combinations(class_combinations, 2):
        # use precalculated or determined classes
        if picked_classes is not None:
            each = picked_classes
            skip_rest = True

        print('iteration with:', each)

        # select the classes
        ix = np.where(np.isin(labels, each))[0]          # holds the indices of the currently studied classes
        selected_features = features[ix, :].copy()
        selected_labels = labels[ix].copy()

        # cross-validation to separate classes
        y_sup = np.zeros(n_samples)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_idx, test_idx in skf.split(selected_features, selected_labels):
            # fit + predict
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5)
            clf.fit(selected_features[train_idx, :], selected_labels[train_idx])
            y_sup[test_idx] = clf.predict(selected_features[test_idx, :])

        # unsupervised classification
        y_unsup = predict_unsupervised('kmeans', selected_features, selected_labels)

        # class index
        ix_c1 = np.where(selected_labels == each[0])[0]
        ix_c2 = np.where(selected_labels == each[1])[0]

        # supervised
        ix_ys_c1 = np.where(y_sup == each[0])[0]
        ix_ys_c2 = np.where(y_sup == each[1])[0]

        # unsupervised
        ix_yu_c1 = np.where(y_unsup == each[0])[0]
        ix_yu_c2 = np.where(y_unsup == each[1])[0]

        # supervised correct + remaining
        ix_ys_c1_correct = np.intersect1d(ix_c1, ix_ys_c1)
        ix_ys_c2_correct = np.intersect1d(ix_c2, ix_ys_c2)
        ix_c1_remaining = np.setdiff1d(ix_c1, ix_ys_c1_correct)
        ix_c2_remaining = np.setdiff1d(ix_c2, ix_ys_c2_correct)

        # unsupervised correct + remaining
        ix_yu_c1_correct = np.intersect1d(ix_ys_c1_correct, ix_yu_c1)
        ix_yu_c1_faulty = np.setdiff1d(ix_ys_c1_correct, ix_yu_c1_correct)
        ix_yu_c2_correct = np.intersect1d(ix_ys_c2_correct, ix_yu_c2)
        ix_yu_c2_faulty = np.setdiff1d(ix_ys_c2_correct, ix_yu_c2_correct)

        assert len(ix) == len(ix_c1) + len(ix_c2), 'Error - selected labels wrong'

        # store
        index_set1 = {0: ix[ix_yu_c1_correct], 1: ix[ix_yu_c1_faulty], 2: ix[ix_c1_remaining]}
        assert len(np.intersect1d(index_set1[0], index_set1[1])) == 0, 'Error in choosing class instances'
        assert len(np.intersect1d(index_set1[0], index_set1[2])) == 0, 'Error in choosing class instances'
        assert len(np.intersect1d(index_set1[1], index_set1[2])) == 0, 'Error in choosing class instances'
        assert np.sum([len(index_set1[i]) for i in range(3)]) == len(ix_c1), 'Error - choosing class instances'

        index_set2 = {0: ix[ix_yu_c2_correct], 1: ix[ix_yu_c2_faulty], 2: ix[ix_c2_remaining]}
        assert len(np.intersect1d(index_set2[0], index_set2[1])) == 0, 'Error in choosing class instances'
        assert len(np.intersect1d(index_set2[0], index_set2[2])) == 0, 'Error in choosing class instances'
        assert len(np.intersect1d(index_set2[1], index_set2[2])) == 0, 'Error in choosing class instances'
        assert np.sum([len(index_set2[i]) for i in range(3)]) == len(ix_c2), 'Error - choosing class instances'

        # confusion
        combo_confusion = len(index_set1[1]) + len(index_set2[1])

        # adapt confusion etc.
        if combo_confusion > confusion:
            # update the confusion metric
            confusion = combo_confusion

            # set the sets of normals and anomalies
            # smallest class is the anomaly class
            if len(index_set1[0]) > len(index_set2[0]):
                normals = index_set1
                anomalies = index_set2
                best_combination = each
            else:
                normals = index_set2
                anomalies = index_set1
                best_combination = (each[1], each[0])

        if skip_rest:
            break

    # print classes picked
    print('Classes picked:', best_combination, 'with', confusion, 'confusion')

    return normals, anomalies, best_combination


def construct_source_sets(src_n1, src_a1, picked_classes, combos, combo_names, features, labels, anomaly_percent, dataset_path, name, version):
    """ Construct the source sets. """

    # construct the source domains
    for ii, each in enumerate(combos):
        print('Source combination: ', combo_names[ii])
        # normals
        n1 = each[0]
        n2 = None
        if type(n1) == tuple:
            n1 = each[0][0]
            n2 = each[0][1]
        if n1 == picked_classes[0]:
            idx = np.concatenate((src_n1[0], src_n1[1]))
            normals1 = features[idx, :]
        else:
            idx = np.where(labels == n1)[0]
            normals1 = features[idx, :]
        if n2 is not None:
            if n2 == picked_classes[0]:
                idx = np.concatenate((src_n1[0], src_n1[1]))
                normals2 = features[idx, :]
            else:
                idx = np.where(labels == n2)[0]
                normals2 = features[idx, :]
        else:
            normals2 = np.array([])

        # anomalies
        a1 = each[1]
        a2 = None
        if type(a1) == tuple:
            a1 = each[1][0]
            a2 = each[1][1]
        if a1 == picked_classes[1]:
            idx = np.concatenate((src_a1[0], src_a1[1]))
            anomalies1 = features[idx, :]
        else:
            idx = np.where(labels == a1)[0]
            anomalies1 = features[idx, :]
        if a2 is not None:
            if a2 == picked_classes[1]:
                idx = np.concatenate((src_a1[0], src_a1[1]))
                anomalies2 = features[idx, :]
            else:
                idx = np.where(labels == a2)[0]
                anomalies2 = features[idx, :]
        else:
            anomalies2 = np.array([])

        # select how many instances
        n1, n2 = len(normals1), len(normals2)
        a1, a2 = len(anomalies1), len(anomalies2)
        pn2 = min(n2, int(0.5 * MIN_SIZE))
        pn1 = min(n1, MIN_SIZE - pn2)

        a_size = int(math.ceil(anomaly_percent * (pn1 + pn2)))
        pa2 = min(a2, int(0.5 * a_size))
        pa1 = min(a1, a_size - pa2)

        print('normals:', pn1, '/', normals1.shape, '-', pn2, '/', normals2.shape, '-', \
              pa1, '/', anomalies1.shape, '-', pa2, '/', anomalies2.shape)

        # select the instances
        idx1 = np.random.choice(n1, pn1, replace=False)
        fnorm1 = normals1[idx1, :]
        if pn2 > 0:
            idx2 = np.random.choice(n2, pn2, replace=False)
            fnorm2 = normals2[idx2, :]
            nn = np.vstack((fnorm1, fnorm2))
        else:
            nn = fnorm1

        idx1 = np.random.choice(a1, pa1, replace=False)
        fanom1 = anomalies1[idx1, :]
        if pa2 > 0:
            idx2 = np.random.choice(a2, pa2, replace=False)
            fanom2 = anomalies2[idx2, :]
            aa = np.vstack((fanom1, fanom2))
        else:
            aa = fanom1

        # combine data and labels
        data = np.vstack((nn, aa))
        temp_labels = np.ones(len(data)) * -1
        temp_labels[-len(aa):] = 1

        # store results
        df = pd.DataFrame(data)
        df['labels'] = temp_labels
        df.to_csv(os.path.join(dataset_path, name + '_source_' + combo_names[ii] + '_v' + str(version) + '.csv'), sep=',')


def construct_data_sets(norm, anom, features, anomaly_percent, dataset_path, name, versions):
    """ Construct and store the datasets. """

    n, a = len(norm[0]) + len(norm[1]), len(anom[0]) + len(anom[1])
    n0, n1 = len(norm[0]), len(norm[1])
    a0, a1 = len(anom[0]), len(anom[1])

    # normals (try sampling equally from each set)
    pn1 = min(n1, max(MIN_SUBSET, int(n1 / n * MIN_SIZE)))
    pn0 = min(n0, MIN_SIZE - pn1)

    # anomalies
    a_size = int(int(math.ceil(anomaly_percent * (pn1 + pn0))))
    pa1 = min(a1, max(int(a_size / 2), int(a1 / a * a_size)))
    pa0 = min(a0, a_size - pa1)

    print('good normals:', pn0, '/', n0, '- bad normals:', pn1, '/', n1)
    print('good anomalies:', pa0, '/', a0, '- bad anomalies:', pa1, '/', a1)

    for i in range(versions):
        # normals
        if n0 > 0:
            ix = norm[0][np.random.choice(n0, pn0, replace=False)]
            normals0 = features[ix, :]
        else:
            normals0 = np.array([])
        if n1 > 0:
            ix = norm[1][np.random.choice(n1, pn1, replace=False)]
            normals1 = features[ix, :]
        else:
            normals1 = np.array([])

        # anomalies
        if a0 > 0:
            ix = anom[0][np.random.choice(a0, pa0, replace=False)]
            anomalies0 = features[ix, :]
        else:
            anomalies0 = np.array([])
        if a1 > 0:
            ix = anom[1][np.random.choice(a1, pa1, replace=False)]
            anomalies1 = features[ix, :]
        else:
            anomalies1 = np.array([])

        # combine
        if len(normals1) > 0:
            nn = np.vstack((normals0, normals1))
        else:
            nn = normals0
        if len(anomalies1) > 0:
            aa = np.vstack((anomalies0, anomalies1))
        else:
            aa = anomalies0

        # combine data and labels
        data = np.vstack((nn, aa))
        labels = np.ones(len(data)) * -1
        labels[-len(aa):] = 1

        # store results
        df = pd.DataFrame(data)
        df['labels'] = labels
        df.to_csv(os.path.join(dataset_path, name + '_v' + str(i) + '.csv'), sep=',')


def predict_unsupervised(method, features, labels):
    """ Predict the class of the instances. """

    if method == 'kmeans':
        # elbow method to select number of clusters
        mean_dist = []
        for k in [5, 10, 15, 20, 25, 30]:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(features)
            mean_dist.append(kmeans.inertia_ / features.shape[0])

        # Elbow: explain x % of the variance
        explained_var = 0.95
        variance = np.diff(mean_dist) * -1
        distortion_percent = np.cumsum(variance) / (mean_dist[0] - mean_dist[-1])
        idx = np.where(distortion_percent > explained_var)[0]
        best_k = (idx[0] + 1) * 5

        # kmeans clustering
        kmeans = KMeans(n_clusters=best_k)
        kmeans.fit(features)
        y_pred = kmeans.predict(features)  # cluster labels

    elif method == 'gmm':
        pass

    # determine the class label of each cluster
    l1, l2 = np.unique(labels)
    class_1 = np.zeros(best_k)
    class_2 = np.zeros(best_k)
    for i, cl in enumerate(y_pred):
        if labels[i] == l1:
            class_1[cl] += 1
        else:
            class_2[cl] += 1

    class_dict = dict()
    for i in range(best_k):
        if class_1[i] > class_2[i]:
            class_dict[i] = l1
        else:
            class_dict[i] = l2

    # predict the labels of individual instances
    y_pred_final = np.zeros(len(labels))
    for i, l in enumerate(y_pred):
        y_pred_final[i] = class_dict[l]

    return y_pred_final


def load_data(dataset_path, drop_classes=[]):
    """ Load data from file. """
    data = pd.read_csv(dataset_path, sep=',').iloc[:, 1:]
    features = data.iloc[:, :-1].values.astype(float)
    labels = data.iloc[:, -1].values
    for c in drop_classes:
        idx = np.where(labels != c)[0]
        features = features[idx, :]
        labels = labels[idx]

    # z-normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # classes to: 0, 1 ... n
    classes_ = np.unique(labels)
    class_dict = dict()
    for i, c in enumerate(classes_):
        class_dict[c] = i
    ncl = i + 1

    # change the label vector
    labels = np.array([class_dict[l] for l in labels])
    classes_ = np.array([i for i in range(ncl)])

    return features, scaled_features, labels, ncl


def subsample_data(features, scaled_features, labels, subsamp):
    """ Subsample the data. """

    for k, v in subsamp.items():
        ix = np.where(labels == k)[0]
        ix_rest = np.where(labels != k)[0]
        sample_ix = np.random.choice(ix, int(v * len(ix)), replace=False)
        keep_ix = np.union1d(ix_rest, sample_ix)
        # subsample
        features = features[keep_ix, :]
        scaled_features = scaled_features[keep_ix, :]
        labels = labels[keep_ix]

    return features, scaled_features, labels


# -------------
# RUN SCRIPT
# -------------

if __name__ == '__main__':
    sys.exit(main())
