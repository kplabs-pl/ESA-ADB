# Simple standard deviation thresholding

## Parameters

- `tol`: float >= 0, optional (default 3.0)  
  Number of standard deviation above/below the mean to treat as an anomaly

- `random_state`: int, RandomState instance or None, optional (default None)  
  If int, random_state is the seed used by the random number generator;
  If RandomState instance, random_state is the random number generator;
  If None, the random number generator is the RandomState instance used by `np.random`.
  Used when `svd_solver == 'arpack' or svd_solver == 'randomized'`.

