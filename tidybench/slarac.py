"""
Implements the SLARAC (Subsampled Linear Auto-Regression Absolute
Coefficients) algorithm.

Based on an implementation that is originally due to Sebastian Weichwald
(sweichwald).
"""


import numpy as np
from sklearn.utils import resample
from .utils import common_pre_post_processing


INV_GOLDEN_RATIO = 2 / (1 + np.sqrt(5))


@common_pre_post_processing
def slarac(data,
           maxlags=1,
           n_subsamples=50,
           subsample_sizes=[INV_GOLDEN_RATIO**(1 / k) for k in [1, 2, 3, 6]],
           aggregate_lags=lambda x: x.max(axis=1),
           ):
    """SLARAC (Subsampled Linear Auto-Regression Absolute Coefficients).

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

    aggregate_lags : function
        Function that takes an N (to) x maxlags x N (from) ndarray as input and
        outputs an N x N ndarray aggregating the lag-resolved scores,
        for example
            lambda x: x.max(axis=1)
        or
            lambda x: x.sum(axis=1)

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

    # Obtain absolute regression coefficients on the entire data set and
    # random subsamples
    scores = np.abs(varmodel(data, maxlags))
    for subsample_size in np.random.choice(subsample_sizes, n_subsamples):
        n_samples = int(np.round(subsample_size * T))
        scores += np.abs(varmodel(data, maxlags, n_samples=n_samples))

    # Drop the intercepts and divide the sum to obtain the average
    scores = scores[:, 1:] / (n_subsamples + 1)

    # Aggregate lagged coefficients to square connectivity matrix
    scores = aggregate_lags(scores.reshape(N, -1, N)).T
    return scores


def varmodel(data, maxlags=1, n_samples=None):
    # Stack data to perform regression of present on past values
    Y = data.T[:, maxlags:]
    d = Y.shape[0]
    Z = np.vstack([np.ones((1, Y.shape[1]))] +
                  [data.T[:, maxlags - k:-k]
                   for k in range(1, maxlags + 1)])

    # Subsample data
    if n_samples is not None:
        Y, Z = resample(Y.T, Z.T, replace=False, n_samples=n_samples)
        Y, Z = Y.T, Z.T

    # Heuristic to determine a feasible number of lags
    feasiblelag = maxlags
    if Z.shape[1] / Z.shape[0] < INV_GOLDEN_RATIO:
        feasiblelag = int(np.floor(
            (Z.shape[1] / INV_GOLDEN_RATIO - 1) / d))
    # Choose a random effective lag that is feasible and <= maxlag
    efflag = np.random.choice(np.arange(1, max(maxlags, feasiblelag) + 1))
    indcutoff = efflag * d + 1

    # Obtain linear regression coefficients
    B = np.zeros((d, maxlags * d + 1))
    B[:, :indcutoff] = np.linalg.lstsq(
        Z[:indcutoff, :].dot(Z[:indcutoff, :].T),
        Z[:indcutoff, :].dot(Y.T),
        rcond=None)[0].T
    return B
