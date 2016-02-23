from sklearn.preprocessing import label_binarize


def binarize(y, y_min, y_max):
    """

    :param y: input value
    :param y_min: minimum value the class can take
    :param y_max: max value -||-
    :return: binarized vector in {0, ymax - ymin + 1}
             containing 1 on the index = (input value - ymin)
             and zeros everywhere else
    """
    # translating features to a binary vector
    temp = label_binarize([y], classes=range(y_min, y_max + 1))
    return temp
