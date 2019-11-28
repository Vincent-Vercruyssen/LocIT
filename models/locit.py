# -*- coding: UTF-8 -*-
"""

Full LocIT (with SSkNNO) algorithm.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.neighbor import BallTree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# ----------------------------------------------------------------------------
# LocIT + SSkNNO
# ----------------------------------------------------------------------------

def apply_LocIT(Xs, Xt, ys=None, yt=None,
        psi=10, train_selection='random', scaling=True,
        k=10, supervision='loose'):
    """ Apply LocIT + SSkNNO.

    Parameters
    ----------
    Xs : np.array of shape (n_samples, n_features), optional (default=None)
        The source instances.
    Xt : np.array of shape (n_samples, n_features), optional (default=None)
        The target instances.
    ys : np.array of shape (n_samples,), optional (default=None)
        The ground truth of the source instances.
    yt : np.array of shape (n_samples,), optional (default=None)
        The ground truth of the target instances.
    
    psi : int (default=10)
        Neighborhood size.
    
    train_selection : str (default='random')
        How to select the negative training instances:
        'farthest'  --> select the farthest instance
        'random'    --> random instance selected
        'edge'      --> select the (psi+1)'th instance
    
    scaling : bool (default=True)
        Scale the source and target domain before transfer.

    k : int (default=10)
        Number of nearest neighbors.
    
    supervision : str (default=loose)
        How to compute the supervised score component.
        'loose'     --> use all labeled instances in the set of nearest neighbors
        'strict'    --> use only instances that also count the instance among their neighbors
    
    Returns
    -------
    yt_scores : np.array of shape (n_samples,)
        Anomaly scores for the target instances.
    """

    # input
    if ys is None:
        ys = np.zeros(Xs.shape[0])
    Xs, ys = check_X_y(Xs, ys)
    if yt is None:
        yt = np.zeros(Xt.shape[0])
    Xt, yt = check_X_y(Xt, yt)

    # scaling
    if scaling:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xt)

    # transfer
    ixt = _instance_transfer(Xs, Xt, ys, psi, train_selection)
    Xs_trans = Xs[ixt, :]
    ys_trans = ys[ixt]

    # combine
    X_combo = np.vstack((Xs_trans, Xt))
    y_combo = np.zeros(X_combo.shape[0], dtype=int)
    y_combo[:len(ys_trans)] = ys_trans

    # anomaly detection
    source_contamination = len(np.where(ys > 0)[0]) / len(np.where(ys != 0)[0])

    all_scores = _ssknno_anomaly_detection(X_combo, y_combo,
        source_contamination, k, supervision)
    yt_scores = all_scores[len(ys_trans):]

    return yt_scores

def _ssknno_anomaly_detection(X, y, c, k, supervision):
    """ Do the SSkNNO detection. """

    tol = 1e-10
    n, _ = X.shape

    # construct the BallTree
    tree = BallTree(X, leaf_size=16, metric='euclidean')
    D, _ = tree.query(X, k=k+1)

    # compute gamma
    outlier_score = D[:, -1].flatten()
    gamma = np.percentile(outlier_score, int(100 * (1.0 - c))) + tol

    # labels and radii
    labels = y.copy()
    radii = D[:, -1].flatten() + tol

    # 

    return None

def _instance_transfer(Xs, Xt, ys, psi, train_selection):
    """ Do the instance transfer. """

    tol = 1e-10

    ns, _ = Xs.shape
    nt, _ = Xt.shape

    # neighbor trees
    target_tree = BallTree(Xt, leaf_size=16, metric='euclidean')
    source_tree = BallTree(Xs, leaf_size=16, metric='euclidean')

    # 1. construct the transfer classifier
    _, Ixs = target_tree.query(Xt, k=nt)

    X_train = np.zeros((2 * nt, 2), dtype=np.float)
    y_train = np.zeros(2 * nt, dtype=np.float)
    random_ixs = np.arange(0, nt, 1)
    np.random.shuffle(random_ixs)

    for i in range(nt):
        # local mean and covaraiance matrix of the current point
        NN_x = Xt[Ixs[i, 1:psi+1], :]
        mu_x = np.mean(NN_x, axis=0)
        C_x = np.cov(NN_x.T)

        # POS: local mean and covariance matrix of the nearest neighbor
        nn_ix = Ixs[i, 1]
        NN_nn = Xt[Ixs[nn_ix, 1:psi+1], :]
        mu_nn = np.mean(NN_nn, axis=0)
        C_nn = np.cov(NN_nn.T)

        # NEG: local mean and covariance matrix of a randomly selected point
        if train_selection == 'random':
            r_ix = random_ixs[i]
        elif train_selection == 'edge':
            r_ix = Ixs[i, psi+2]
        elif train_selection == 'farthest':
            r_ix = Ixs[i, -1]
        else:
            raise ValueError(train_selection,
                'not valid!')
        NN_r = Xt[Ixs[r_ix, 1:psi], :]
        mu_r = np.mean(NN_r, axis=0)
        C_r = np.cov(NN_r.T)

        # training vectors
        f_pos = np.array([float(np.linalg.norm(mu_x - mu_nn)), float(
            np.linalg.norm(C_x - C_nn)) / float(np.linalg.norm(C_x) + tol)])
        f_neg = np.array([float(np.linalg.norm(mu_x - mu_r)), float(
            np.linalg.norm(C_x - C_r)) / float(np.linalg.norm(C_x) + tol)])

        X_train[2*i, :] = f_pos
        y_train[2*i] = 1.0
        X_train[2*i+1, :] = f_neg
        y_train[2*i+1] = 0.0

    X_train = np.nan_to_num(X_train)
    transfer_scaler = StandardScaler()
    X_scaled = transfer_scaler.fit_transform(X_train)

    clf = _optimal_transfer_classifier(X_scaled, y_train)

    # 2. determine which source instances to transfer
    Xs_feat = np.zeros((ns, 2), dtype=np.float)
    _, Ixs = source_tree.query(Xs, k=psi+1)
    _, Ixt = target_tree.query(Xs, k=psi+1)

    for i in range(ns):
        # local mean and covariance matrix in the source domain transfer_ixs
        NN_s = Xs[Ixs[i, 1:psi+1], :]  # nearest neighbors in the source domain
        mu_s = np.mean(NN_s, axis=0)
        C_s = np.cov(NN_s.T)

        # local mean and covariance matrix in the target domain
        NN_t = Xt[Ixt[i, :psi], :]  # nearest neighbors in the target domain
        mu_t = np.mean(NN_t, axis=0)
        C_t = np.cov(NN_t.T)

        f = np.array([float(np.linalg.norm(mu_s - mu_t)), float(
            np.linalg.norm(C_s - C_t)) / float(np.linalg.norm(C_s) + tol)])
        Xs_feat[i, :] = f

    Xs_feat = np.nan_to_num(Xs_feat)
    Xs_scaled = transfer_scaler.transform(Xs_feat)

    transfer_labels = clf.predict(Xs_scaled)
    ixt = np.where(transfer_labels == 1.0)[0]

    return ixt

def _optimal_transfer_classifier(train, labels):
    """ optimal transfer classifier based on SVC """

    # tuning parameters
    tuned_parameters = [{'C': [0.01, 0.1, 0.5, 1, 10, 100],
                        'gamma': [0.01, 0.1, 0.5, 1, 10, 100],
                        'kernel': ['rbf']},
                        {'kernel': ['linear'],
                        'C': [0.01, 0.1, 0.5, 1, 10, 100]}]
    
    # grid search
    svc = SVC(probability=True)
    clf = GridSearchCV(svc, tuned_parameters, cv=3, refit=True)
    clf.fit(train, labels)
    
    # return classifier
    return clf

def _squashing_function(x, p):
    """ Compute the value of x under squashing function with parameter p. """
    
    return 1.0 - np.exp(np.log(0.5) * np.power(x / p, 2))
