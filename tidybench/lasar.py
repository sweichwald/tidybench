"""
Implements the LASAR (LASso Auto-Regression) algorithm.

Based on an implementation that is originally due to Sebastian Weichwald
(sweichwald).
"""


import numpy as np
from sklearn.linear_model import LassoLarsCV
from sklearn.utils import resample
from .utils import commonpreprocessing, commonpostprocessing


INV_GOLDEN_RATIO = 2 / (1 + np.sqrt(5))


def lasar(data,
          normalise=True,
          maxlags=1,
          speedup=False,
          aggregatelagmax=False,
          normalise_data=False,
          standardise_scores=False):
    data = commonpreprocessing(data,
                               normalise_data=normalise_data)

    lags = maxlags

    timeconsecutivebootstrap = False

    # T timepoints, N variables
    T, N = data.shape

    noofshifts = 123
    if speedup:
        noofshifts = 62

    scores = np.abs(lassovar(data, lags))
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
                    lassovar(data[shift:shift + samples, :], lags))
        else:
            for _ in range(noofshifts):
                scores += np.abs(lassovar(data, lags, n_samples=samples))

    # aggregate lagged coefficients to square connectivity matrix
    if aggregatelagmax:
        scores = np.abs(scores.reshape(N, -1, N)).max(axis=1).T
    else:
        scores = np.abs(scores.reshape(N, -1, N)).sum(axis=1).T

    scores = commonpostprocessing(scores,
                                  standardise_scores=standardise_scores)
    return scores


def lassovar(data, lag=1, n_samples=None):
    Y = data.T[:, lag:]
    d = Y.shape[0]
    Z = np.vstack([data.T[:, lag - k:-k]
                   for k in range(1, lag + 1)])
    Y, Z = Y.T, Z.T
    if n_samples is not None:
        Y, Z = resample(Y, Z, replace=False, n_samples=n_samples)

    scores = np.zeros((d, d * lag))

    ls = LassoLarsCV(cv=10, n_jobs=1)

    residuals = np.zeros(Y.shape)

    # one variable after the other as target
    for j in range(d):
        target = np.copy(Y[:, j])
        selectedparents = np.full(d * lag, False)
        # we include one lag after the other
        for l in range(1, lag + 1):
            ind_a = d * (l - 1)
            ind_b = d * l
            ls.fit(Z[:, ind_a:ind_b], target)
            selectedparents[ind_a:ind_b] = ls.coef_ > 0
            target -= ls.predict(Z[:, ind_a:ind_b])

        residuals[:, j] = np.copy(target)

        # refit to get rid of the bias
        ZZ = Z[:, selectedparents]
        B = np.linalg.lstsq(ZZ.T.dot(ZZ), ZZ.T.dot(Y[:, j]), rcond=None)[0]
        scores[j, selectedparents] = B

    # the more uncorrelated the residuals the higher the weight
    weight = 1
    res = np.corrcoef(residuals.T)
    if np.linalg.matrix_rank(res) == res.shape[0]:
        weight = np.linalg.det(res)
    return scores * weight
