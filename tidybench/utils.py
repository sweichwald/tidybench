import numpy as np


def common_pre_post_processing(func_raw):
    """
    Used as decorator to add common pre-processing steps of the data (args[0])
    and common post-processing steps of the scores (out[0]) to an algorithm.

    pre_normalise: boolean; whether to normalise the data
    post_standardise: boolean; whether to standardise the scores
    """
    def func(*args, **kwargs):
        pre_normalise = kwargs.pop("pre_normalise", False)
        post_standardise = kwargs.pop("post_standardise", False)

        if pre_normalise:
            args = list(args)
            args[0] = standardise(args[0])
            args = tuple(args)

        out = func_raw(*args, **kwargs)

        if post_standardise:
            if type(out) == list and len(out) > 1:
                out[0] = standardise(out[0], axis=None)
            else:
                out = standardise(out, axis=None)

        return out

    return func


def standardise(X, axis=0, keepdims=True, copy=False):
    if copy:
        X = np.copy(X)
    X -= X.mean(axis=axis, keepdims=keepdims)
    X /= X.std(axis=axis, keepdims=keepdims)
    return X
