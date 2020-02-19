"""
Implements the QRBS (Quantiles of Ridge regressed Bootstrap Samples) algorithm.

Based on an implementation that is originally due to Nikolaj Thams
(nikolajthams).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils import resample


def handle_nans(data):
    # Function for linearly interpolating NAs
    def interpnans(y):
        nans = np.isnan(y)
        def x(z): return z.nonzero()[0]
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y

    # Re-code NAs and linearly interpolate
    data[data == 999] = np.nan
    for j in range(data.shape[1]):
        data[:, j] = interpnans(data[:, j])
    return data


def ridge(data, lags=1, alpha=.005, q=.75, normalize=False, n_resamples=600,
          experiment=None):
    """
    Perform bootstrapped ridge regression of data at time t on data in the past

    Parameters
    ----------
    lags : int
        Number of lags to include in the modelling

    alpha : double
        Penalization parameter used for the ridge regression

    q : double
        The method performs 200 bootstrap samples, in each fitting a ridge
        regression on a random subset of the data. This gives 200 estimates
        of the effect i -> j.
        We take the q'th quantile as the final estimate.
        q = 1 corresponds to the max effect across samples, q = 0.5 to the
        median effect.

    normalize : boolean
        Whether or not the data should be pre-normalized.

    n_resamples : int
        Number of bootstrap samples drawn

    experiment : str
        The experiment name (e.g. TestCLIM_N-5_T-100).
        Only used for automatically handling missing data in the
        TestWEATHmiss datasets

    Returns
    ----------
    scores : ndarray
        Array with scores for each link i -> j
    """
    if 'miss' in experiment:
        data = handle_nans(data)

    # Normalize the data to mean 0 and unit variance
    if normalize:
        data -= data.mean(axis=0)
        data /= np.sqrt(np.var(data, axis=0))

    # We regress y = data_t on X = data_[t-1, ..., t-lags]
    y = np.diff(data, axis=0)[lags-1:]
    X = np.concatenate([data[lag:-(lags-lag)]
                        for lag in np.flip(np.arange(lags))], axis=1)

    # Initiate ridge regressor
    ls = Ridge(alpha)

    # Bootstrap fit lasso coefficients
    k = int(np.floor(data.shape[0]*0.7))
    results = np.stack([
        ls.fit(*resample(X, y, n_samples=k)).coef_
        for _ in range(n_resamples)])

    # Aggregate lags by taking abs and summing
    results = np.abs(
        results.reshape(n_resamples, y.shape[1], lags, -1)).sum(axis=2)
    scores = np.quantile(results, q, axis=0)

    # Normalize scores to the [0, 1] interval
    scores -= scores.min()
    scores /= scores.max()
    scores /= scores.mean()

    # Return transposed scores because our format beta*X means you can read
    # parents by row but CauseMe reads parents of i in column i
    return scores.T
