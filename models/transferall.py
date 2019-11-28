# -*- coding: UTF-8 -*-
"""

Transferall.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# ----------------------------------------------------------------------------
# Transferall
# ----------------------------------------------------------------------------

def apply_transferall(Xs, Xt, ys=None, yt=None, scaling=True, k=10):
    """ Apply Transferall.

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
    
    k : int (default=10)
        Number of nearest neighbors.

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

    # anomaly detection (kNN) - fit on source
    X_combo = np.vstack((Xs, Xt))
    y_combo = np.zeros(X_combo.shape[0], dtype=int)
    y_combo[:len(ys)] = ys

    yt_scores = _kNN_anomaly_detection(X_combo, y_combo, Xt, k)
    
    return yt_scores


def _kNN_anomaly_detection(X, y, Xt, k):
    """ Apply kNN anomaly detection. """

    ixl = np.where(y != 0)[0]
    Xtr = X[ixl, :]
    ytr = y[ixl]
    
    # fit
    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean', algorithm='ball_tree')
    clf.fit(Xtr, ytr)

    # predict
    yt_scores = clf.predict_proba(Xt)
    if len(clf.classes_) > 1:
        ix = np.where(clf.classes_ == 1)[0][0]
        yt_scores = yt_scores[:, ix].flatten()
    else:
        yt_scores = yt_scores.flatten()

    return yt_scores