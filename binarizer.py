import numpy as np

from sklearn.preprocessing import label_binarize


def binarize(y):
    """

    :param y: input vector of containing range of values
    :return: binarized vector of size = y_max - y_min + 1
             which contains 1 on the index of the y value
             and zeros everwhere else
    """

    # translating features to a binary vector
    y_min = np.min(y)
    y_max = np.max(y)
    temp = label_binarize(y, classes=range(y_min, y_max + 1))

    return temp
