"""
Implements the SLARAC (Subsampled Linear Auto-Regression Absolute Coefficients) and LASAR (LASso Auto-Regression) algorithms.

Based on an implementation that is originally due to Sebastian Weichwald (sweichwald).
"""


import numpy as np
from sklearn.linear_model import LassoLarsCV
from sklearn.utils import resample


INV_GOLDEN_RATIO = 2 / (1 + np.sqrt(5))


def varvar(data,
           differences=False,
           normalise=True,
           maxlags=1,
           speedup=False,
           edgeprior=False,
           aggregatelagmax=False,
           zeroonescaling=True,
           lasso=False,
           **kwargscatchall):
    lags = maxlags

    timeconsecutivebootstrap = False

    if differences:
        data = np.diff(data, axis=0)

    if normalise:
        data -= data.mean(axis=0, keepdims=True)
        data /= data.std(axis=0, keepdims=True)

    # T timepoints, N variables
    T, N = data.shape

    noofshifts = 123
    if speedup:
        noofshifts = 62

    if not lasso:
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

    else:
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

    if zeroonescaling:
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    if edgeprior:
        scores /= np.mean(scores.reshape(-1))

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
