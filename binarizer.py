import numpy as np
from sklearn.preprocessing import label_binarize


def binarize(y):
    # translating features to a binary vector
    min = np.min(y)
    max = np.max(y)
    temp = label_binarize(y, classes=range(min, max + 1))

    assert isinstance(temp, object)
    return temp
