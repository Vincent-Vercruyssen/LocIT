# -*- coding: UTF-8 -*-
"""

Run experiments.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import sys, os, time, argparse
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

# transfer models
from models.locit import apply_LocIT
from models.transferall import apply_transferall
from models.coral import apply_CORAL

# anomaly detection
from models.knno import apply_kNNO
from models.iforest import apply_iForest


# ----------------------------------------------------------------------------
# run experiment
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run transfer learning - anomaly detection experiment')
    parser.add_argument('-d', '--dataset', type=str, default='', help='dataset = folder in data/ directory')
    parser.add_argument('-m', '--method', type=str, default='', help='method to use')
    args, unknownargs = parser.parse_known_args()
    
    # difficulty dictionary
    transfer_difficulty = {
        'n1_a1': 1,
        'n1_a2': 2,
        'n2_a1': 4,
        'n12_a1': 3,
        'n12_a12': 3,
        'n2_a2': 5}

    # load the data
    main_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(main_path, 'data', args.dataset)
    print('The experiments are executed on the `shuttle` data')

    source_sets, target_sets = _load_and_preprocess_data(data_path)

    # apply algorithms - every combination of source and target
    auc_results = dict()
    dataset_name = ''
    for tgt_name, target_data in target_sets.items():
        dataset_name = tgt_name.split('_v')[0]

        # target data
        Xt = target_data.iloc[:, :-1].values
        yt = target_data.iloc[:, -1].values
        ixtl = np.where(yt != 0.0)[0]
        nt, _ = Xt.shape

        # transfer from each source domain
        for src_name, source_data in source_sets.items():
            # source data
            Xs = source_data.iloc[:, :-1].values
            ys = source_data.iloc[:, -1].values
            ns, _ = Xs.shape

            # actual transfer + anomaly detection
            # TRANSFER METHODS
            if args.method.lower() == 'locit':
                target_scores = apply_LocIT(Xs, Xt.copy(), ys, yt.copy(),
                    k=10, psi=20, scaling=False, supervision='loose',
                    train_selection='farthest')

            elif args.method.lower() == 'transferall':
                target_scores = apply_transferall(Xs, Xt.copy(), ys, yt.copy(),
                    k=10, scaling=True)

            elif args.method.lower() == 'coral':
                target_scores = apply_CORAL(Xs, Xt.copy(), ys, yt.copy(),
                    scaling=True)

            # UNSUPERVISED ANOMALY DETECTION METHODS
            elif args.method.lower() == 'knno':
                target_scores = apply_kNNO(Xs, Xt.copy(), ys, yt.copy(), scaling=False)

            elif args.method.lower() == 'iforest':
                target_scores = apply_iForest(Xs, Xt.copy(), ys, yt.copy(),
                    n_estimators=100, contamination=0.1)

            else:
                raise ValueError(args.method,
                    'is not an implemented/accepted method')

            # compute AUC
            auc = roc_auc_score(y_true=yt, y_score=target_scores)
            print('Transfer:  ', src_name, '\t-->\t', tgt_name, '\tAUC =', auc)

            # store the results
            sn = src_name.split('source_')[1].split('_v')[0]
            if sn in auc_results.keys():
                auc_results[sn].append(auc)
            else:
                auc_results[sn] = [auc]

    # print results
    print('\n\nAUC results on {}:'.format(dataset_name.upper()))
    print('----------------'+'-'*len(dataset_name))
    for k, v in auc_results.items():
        print('  Difficulty level {}: \t{}'.format(transfer_difficulty[k], np.mean(v)))
    print('\nDone!\n')


def _load_and_preprocess_data(data_path):
    """ Load and preprocess the data. """

    src_path = os.path.join(data_path, 'source')
    tgt_path = os.path.join(data_path, 'target')

    # source files
    source_files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    source_files = [os.path.join(src_path, f) for f in source_files if '.csv' in f]

    # target files
    target_files = [f for f in os.listdir(tgt_path) if os.path.isfile(os.path.join(tgt_path, f))]
    target_files = [os.path.join(tgt_path, f) for f in target_files if '.csv' in f]

    # load the data
    source_sets = dict()
    for sf in source_files:
        data = pd.read_csv(sf, sep=',', index_col=0).sample(frac=1).reset_index(drop=True)
        file_name = os.path.split(sf)[1].split('.csv')[0]
        source_sets[file_name] = data

    target_sets = dict()
    for sf in target_files:
        data = pd.read_csv(sf, sep=',', index_col=0).sample(frac=1).reset_index(drop=True)
        file_name = os.path.split(sf)[1].split('.csv')[0]
        target_sets[file_name] = data

    return source_sets, target_sets


if __name__ == '__main__':
    main()
    