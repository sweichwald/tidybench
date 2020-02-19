import numpy as np
import tidybench


if __name__ == "__main__":
    """
    Generate time series data over three variables from a stable SVAR with
    the following structure:
    """
    # --> X_1(t-1) --> X_1(t) --> X_1(t+1) -->
    #              \         \
    #               \         \
    #                \         \
    #                 \         \
    #                  v         v
    # --> X_2(t-1) --> X_2(t) --> X_2(t+1) -->
    #              \   ^     \   ^
    #               \ /       \ /
    #                \         /
    #               / \       / \
    #              /   v     /   v
    # --> X_3(t-1) --> X_3(t) --> X_3(t+1) -->
    B = np.asarray([[1, 2, 0],
                    [0, 1, 1],
                    [0, 2, 1]]) / 3
    T, d = 500, 3
    X = np.random.randn(T, d)
    for t in range(1, T):
        X[t, :] += B.T.dot(X[t-1, :])

    # The true adjacency matrix is
    A = B > 0
    print('True adjacency matrix:')
    print(A)

    print('Score matrix for the adjacency matrix as inferred by '
          'slarac (post_standardised):')
    print(tidybench.slarac(X, post_standardise=True).round(2))

    print('Score matrix for the adjacency matrix as inferred by '
          'qrbs (post_standardised):')
    print(tidybench.qrbs(X, post_standardise=True).round(2))

    print('Score matrix for the adjacency matrix as inferred by '
          'lasar (post_standardised):')
    print(tidybench.lasar(X, post_standardise=True).round(2))
