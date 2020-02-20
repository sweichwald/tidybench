"""
Implements the LASAR (LASso Auto-Regression) algorithm.

Based on an implementation that is originally due to Sebastian Weichwald
(sweichwald).
"""


import numpy as np
from sklearn.linear_model import LassoLarsCV
from sklearn.utils import resample
from .utils import common_pre_post_processing


INV_GOLDEN_RATIO = 2 / (1 + np.sqrt(5))


@common_pre_post_processing
def lasar(data,
          maxlags=1,
          n_subsamples=50,
          subsample_sizes=[INV_GOLDEN_RATIO**(1 / k) for k in [1, 2, 3, 6]],
          cv=5,
          aggregate_lags=lambda x: x.max(axis=1).T,
          ):
    """LASAR (LASso Auto-Regression).

    Parameters
    ----------
    data : ndarray
        T (timepoints) x N (variables) input data

    maxlags : int
        Maximum number of lags to consider

    n_subsamples : int
        How often to subsample the data

    subsample_sizes : ndarray
        Possible sizes of the subsamples as fractions

    cv : int
        Number of cross-validation folds for the lasso variable selection step

    aggregate_lags : function
        Function that takes an N (to) x maxlags x N (from) ndarray as input and
        outputs an N x N ndarray aggregating the lag-resolved scores,
        for example
            lambda x: x.max(axis=1).T
        or
            lambda x: x.sum(axis=1).T

    Arguments for the common pre-processing steps of the data and the common
    post-processing steps of the scores are documented in
    utils.common_pre_post_processing

    Returns
    ----------
    scores : ndarray
        Array where the (i,j)th entry corresponds to the link X_i --> X_j
    """

    # T timepoints, N variables
    T, N = data.shape

    # Obtain absolute regression coefficients after refitting on a cross-
    # validated variable selection obtained by lasso regression on the entire
    # data set and random subsamples
    scores = np.abs(lassovar(data, maxlags, cv=cv))
    for subsample_size in np.random.choice(subsample_sizes, n_subsamples):
        n_samples = int(np.round(subsample_size * T))
        scores += np.abs(lassovar(data, maxlags, n_samples=n_samples, cv=cv))

    # Divide the sum to obtain the average
    scores /= (n_subsamples + 1)

    # aggregate lagged coefficients to square connectivity matrix
    scores = aggregate_lags(scores.reshape(N, -1, N))
    return scores


def lassovar(data, maxlags=1, n_samples=None, cv=5):
    # Stack data to perform regression of present on past values
    Y = data.T[:, maxlags:]
    d = Y.shape[0]
    Z = np.vstack([data.T[:, maxlags - k:-k]
                   for k in range(1, maxlags + 1)])
    Y, Z = Y.T, Z.T

    # Subsample data
    if n_samples is not None:
        Y, Z = resample(Y, Z, replace=False, n_samples=n_samples)

    scores = np.zeros((d, d * maxlags))

    ls = LassoLarsCV(cv=cv, n_jobs=1)

    residuals = np.zeros(Y.shape)

    # Consider one variable after the other as target
    for j in range(d):
        target = np.copy(Y[:, j])
        selectedparents = np.full(d * maxlags, False)
        # Include one lag after the other
        for l in range(1, maxlags + 1):
            ind_a = d * (l - 1)
            ind_b = d * l
            ls.fit(Z[:, ind_a:ind_b], target)
            selectedparents[ind_a:ind_b] = ls.coef_ > 0
            target -= ls.predict(Z[:, ind_a:ind_b])

        residuals[:, j] = np.copy(target)

        # Refit OLS using the selected variables to get rid of the bias
        ZZ = Z[:, selectedparents]
        B = np.linalg.lstsq(ZZ.T.dot(ZZ), ZZ.T.dot(Y[:, j]), rcond=None)[0]
        scores[j, selectedparents] = B

    return scores
