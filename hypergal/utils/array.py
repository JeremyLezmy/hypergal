import numpy as np

def restride(arr, binfactor, squeezed=True, flattened=False):
    """
    Rebin ND-array `arr` by `binfactor`.
    Let `arr.shape = (s1, s2, ...)` and `binfactor = (b1, b2, ...)` (same
    length), new shape will be `(s1/b1, s2/b2, ... b1, b2, ...)` (squeezed).\n
    If `binfactor` is an iterable of length < `arr.ndim`, it is prepended
    with 1's.\n
    If `binfactor` is an integer, it is considered as the bin factor for all
    axes.\n
    If `flattened`, the bin axes are explicitely flattened into a single
    axis. Note that this will probably induce a copy of the array.\n
    Bin 2D-array by a factor 2:
    >>> restride(np.ones((6, 8)), 2).shape
    (3, 4, 2, 2)
    Bin 2D-array by a factor 2, with flattening of the last 2 bin axes:
    >>> restride(np.ones((6, 8)), 2, flattened=True).shape
    (3, 4, 4)
    Bin 2D-array by uneven factor (3, 2):
    >>> restride(np.ones((6, 8)), (3, 2)).shape
    (2, 4, 3, 2)
    Bin 3D-array by factor 2 over the last 2 axes, and take bin average:
    >>> q = np.arange(2*4*6).reshape(2, 4, 6)
    >>> restride(q, (2, 2)).mean(axis=(-1, -2))
    array([[[ 3.5,  5.5,  7.5],
    [15.5, 17.5, 19.5]],
    [[27.5, 29.5, 31.5],
    [39.5, 41.5, 43.5]]])
    Bin 3D-array by factor 2, and take bin average:
    >>> restride(q, 2).mean(axis=(-1, -2, -3))
    array([[15.5, 17.5, 19.5],
    [27.5, 29.5, 31.5]])
    .. Note:: for a 2D-array, `restride(arr, (3, 2))` is equivalent to::
    np.moveaxis(arr.ravel().reshape(arr.shape[1]/3, arr.shape[0]/2, 3, 2), 1, 2)

    """

    try:                        # binfactor is list-like
        # Convert binfactor to [1, ...] + binfactor
        binshape = [1] * (arr.ndim - len(binfactor)) + list(binfactor)
    except TypeError:           # binfactor is not list-like
        binshape = [binfactor] * arr.ndim

    assert len(binshape) == arr.ndim, "Invalid bin factor (shape)."
    assert (~np.mod(arr.shape, binshape).astype('bool')).all(), \
        "Invalid bin factor (modulo)."

    # New shape
    rshape = [ d // b for d, b in zip(arr.shape, binshape) ] + binshape
    # New stride
    rstride = [ d * b for d, b in zip(arr.strides, binshape) ] + list(arr.strides)

    rarr = np.lib.stride_tricks.as_strided(arr, rshape, rstride)

    if flattened:               # Flatten bin axes, which may induce a costful copy!
        rarr = rarr.reshape(rarr.shape[:-(rarr.ndim - arr.ndim)] + (-1,))

    return rarr.squeeze() if squeezed else rarr  # Remove length-1 axes
