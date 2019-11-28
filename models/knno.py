# -*- coding: UTF-8 -*-
"""

kNNo.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree


# ----------------------------------------------------------------------------
# kNNO
# ----------------------------------------------------------------------------

def apply_kNNO(Xs, Xt, ys=None, yt=None, scaling=True, k=10, contamination=0.1):
    """ Apply kNNO.

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
    tree = BallTree(Xt, leaf_size=16, metric='euclidean')
    D, _ = tree.query(Xt, k=k+1)

    # predict
    outlier_scores = D[:, -1].flatten()
    gamma = np.percentile(
        outlier_scores, int(100 * (1.0 - contamination))) + 1e-10
    yt_scores = _squashing_function(outlier_scores, gamma)
    
    return yt_scores


def _squashing_function(x, p):
    """ Compute the value of x under squashing function with parameter p. """
    
    return 1.0 - np.exp(np.log(0.5) * np.power(x / p, 2))
