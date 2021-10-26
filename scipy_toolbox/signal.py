import numpy as np


def full_width_at(x, y, *, p: float = 0.5, k: int = 1):
    """Compute the full width at the p-th fraction of the maximum.

    For `p=0.5` (the default) this corresponds to FWHM.

    Parameters
    ----------
        x : array_like
            The x-coordinates of the data.
        y : array_like
            The y-coordinates of the data.
        p : float
            The fraction of the maximum at which to compute the full width.
        k : int
            The width is computed separately for each of the top-k values in y.

    Returns
    -------
        width : float or list of float
            The full width at each of the k-th top values in y.
            If `k == 1` then the corresponding width is returned.
    """
    top_i = np.argsort(y)[-k:]
    top_y = y[top_i]
    top_w = []
    for i, m in zip(top_i, top_y):
        left = np.interp(p*m, y[:i], x[:i])
        right = np.interp(p*m, y[:i:-1], x[:i:-1])
        top_w.append(right - left)
    if len(top_w) == 1:
        return top_w[0]
    return top_w
