import numpy as np


def common_pre_post_processing(func_raw):
    """
    Used as decorator to add common pre-processing steps of the data (args[0])
    and common post-processing steps of the scores (out[0]) to an algorithm.

    Pre-/post-processing steps are performed in this order, if activated.

    Pre-processing of the data
        pre_normalise: boolean; whether to normalise the data

    Post-processing of the scores
        post_standardise: boolean; whether to standardise the scores
        post_zeroonescaling: boolean; whether to scale the scores to [0, 1]
        post_edgeprior: boolean; whether to divide the scores by their mean
            (may be helpful for comparability of scores across datasets)
    """
    def func(*args, **kwargs):
        pre_normalise = kwargs.pop("pre_normalise", False)

        post_standardise = kwargs.pop("post_standardise", False)
        post_zeroonescaling = kwargs.pop("post_zeroonescaling", False)
        post_edgeprior = kwargs.pop("post_edgeprior", False)

        # Pre-process the data
        if pre_normalise:
            args = list(args)
            args[0] = standardise(args[0])
            args = tuple(args)

        # Call original algorithm
        out = func_raw(*args, **kwargs)

        # Post-process the scores (remaining outputs remain unchanged)
        if type(out) == tuple and len(out) > 1:
            scores = out[0]
        else:
            scores = out

        if post_standardise:
            scores = standardise(scores, axis=None)
        if post_zeroonescaling:
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        if post_edgeprior:
            scores /= scores.mean()

        if type(out) == tuple and len(out) > 1:
            out = list(out)
            out[0] = scores
            out = tuple(out)
        else:
            out = scores

        return out

    return func


def standardise(X, axis=0, keepdims=True, copy=False):
    if copy:
        X = np.copy(X)
    X -= X.mean(axis=axis, keepdims=keepdims)
    X /= X.std(axis=axis, keepdims=keepdims)
    return X
