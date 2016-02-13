import numpy as np
from sklearn.preprocessing import label_binarize


def binarize(y):
    # translating features to a binary vector
    y_min = np.min(y)
    y_max = np.max(y)
    temp = label_binarize(y, classes=range(y_min, y_max + 1))

    return temp
