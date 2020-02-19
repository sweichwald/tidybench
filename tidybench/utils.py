def commonpreprocessing(data,
                        normalise_data=False):
    if normalise_data:
        data -= data.mean(axis=0, keepdims=True)
        data /= data.std(axis=0, keepdims=True)

    return data


def commonpostprocessing(scores,
                         standardise_scores=False):
    if standardise_scores:
        scores -= scores.mean()
        scores /= scores.std()

    return scores
