# -*- coding: UTF-8 -*-
"""

Isolation Forest.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ----------------------------------------------------------------------------
# iForest
# ----------------------------------------------------------------------------

def apply_iForest(Xs, Xt, ys=None, yt=None, scaling=True,
        n_estimators=100, contamination=0.1):
    """ Apply iForest.

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
    
    n_estimators : int (default=100)
        Number of estimators in the ensemble.

    contamination : float (default=0.1)
        The expected contamination in the data.

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
        Xt = scaler.fit_transform(Xt)

    # no transfer!

    # fit 
    clf = IsolationForest(n_estimators=n_estimators,
                          contamination=contamination,
                          behaviour='new',
                          n_jobs=1)
    clf.fit(Xt)

    # predict
    yt_scores = clf.decision_function(Xt) * -1
    yt_scores = (yt_scores - min(yt_scores)) / (max(yt_scores) - min(yt_scores))
    
    return yt_scores
