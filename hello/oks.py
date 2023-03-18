# https://stackoverflow.com/a/73470584/8964990

import numpy as np


def oks(y_true, y_pred, visibility):
    # You might want to set these global constant
    # outside the function scope
    KAPPA = np.array([1] * len(y_true))
    # The object scale
    # You might need a dynamic value for the object scale
    SCALE = 1.0

    # Compute the L2/Euclidean Distance
    distances = np.linalg.norm(y_pred - y_true, axis=-1)
    # Compute the exponential part of the equation
    exp_vector = np.exp(-(distances**2) / (2 * (SCALE**2) * (KAPPA**2)))
    # The numerator expression
    numerator = np.dot(exp_vector, visibility.astype(bool).astype(int))
    # The denominator expression
    denominator = np.sum(visibility.astype(bool).astype(int))
    return numerator / denominator


if __name__ == "__main__":
    IMAGE_SIZE_IN_PIXEL = 50
    gt = (np.random.random((17, 2)) * IMAGE_SIZE_IN_PIXEL).astype(int)
    pred = (np.random.random((17, 2)) * IMAGE_SIZE_IN_PIXEL).astype(int)
    visibility = (np.random.random((17, 1)) * 3).astype(int)

    # On this example the value will not be correct
    # since you need to calibrate KAPPA and SCALE
    print("OKS", oks(gt, pred, visibility))
