"""
Implements the SLARAC (Subsampled Linear Auto-Regression Absolute
Coefficients) algorithm.

Based on an implementation that is originally due to Sebastian Weichwald
(sweichwald).
"""


import numpy as np
from sklearn.utils import resample
from .utils import common_preprocessing, common_postprocessing


INV_GOLDEN_RATIO = 2 / (1 + np.sqrt(5))


def slarac(data,
           maxlags=1,
           speedup=False,
           aggregatelagmax=False,
           normalise_data=False,
           standardise_scores=False):
    data = common_preprocessing(data,
                               normalise_data=normalise_data)

    lags = maxlags

    timeconsecutivebootstrap = False

    # T timepoints, N variables
    T, N = data.shape

    noofshifts = 123
    if speedup:
        noofshifts = 62

    scores = np.abs(varmodel(data, lags))
    Ps = [INV_GOLDEN_RATIO] + \
        [INV_GOLDEN_RATIO**(1 / k) for k in [2, 3, 6]]
    if speedup:
        Ps = [INV_GOLDEN_RATIO**(1 / k) for k in [2, 3]]
    for samples_p in Ps:
        samples = int(np.round(samples_p * T))
        if timeconsecutivebootstrap:
            shifts = np.arange(T - samples + 1)
            if len(shifts > noofshifts):
                shifts = np.random.permutation(shifts)[:noofshifts]
            for shift in shifts:
                scores += np.abs(
                    varmodel(data[shift:shift + samples, :], lags))
        else:
            for _ in range(noofshifts):
                scores += np.abs(varmodel(data, lags, n_samples=samples))
    scores = scores[:, 1:]

    # aggregate lagged coefficients to square connectivity matrix
    if aggregatelagmax:
        scores = np.abs(scores.reshape(N, -1, N)).max(axis=1).T
    else:
        scores = np.abs(scores.reshape(N, -1, N)).sum(axis=1).T

    scores = common_postprocessing(scores,
                                  standardise_scores=standardise_scores)
    return scores


def varmodel(data, lag=1, n_samples=None):
    Y = data.T[:, lag:]
    d = Y.shape[0]
    Z = np.vstack([np.ones((1, Y.shape[1]))] +
                  [data.T[:, lag - k:-k]
                   for k in range(1, lag + 1)])

    if n_samples is not None:
        Y, Z = resample(Y.T, Z.T, replace=False, n_samples=n_samples)
        Y, Z = Y.T, Z.T

    # missing value treatment
    keepinds = (np.sum(Y == 999, axis=0) + np.sum(Z == 999, axis=0)) == 0
    Y = Y[:, keepinds]
    Z = Z[:, keepinds]

    # feasible number of lags
    feasiblelag = lag
    if Z.shape[1] / Z.shape[0] < INV_GOLDEN_RATIO:
        feasiblelag = int(np.floor(
            (Z.shape[1] / INV_GOLDEN_RATIO - 1) / d))
    # random effective lag
    efflag = np.random.choice(np.arange(1, max(lag, feasiblelag) + 1))
    indcutoff = efflag * d + 1

    B = np.zeros((d, lag * d + 1))
    B[:, :indcutoff] = np.linalg.lstsq(
        Z[:indcutoff, :].dot(Z[:indcutoff, :].T),
        Z[:indcutoff, :].dot(Y.T),
        rcond=None)[0].T

    # the more uncorrelated the residuals the higher the weight
    weight = 1
    res = np.corrcoef((B.dot(Z) - Y))
    if np.linalg.matrix_rank(res) == res.shape[0]:
        weight = np.linalg.det(res)
    return B * weight
