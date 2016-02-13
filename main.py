import sys

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from binarizer import binarize
from fscore import fscore

## preprocess the input data
df = pd.read_csv("polen_data.csv")

# shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# remove unnecessary columns
del df["Unnamed: 0"], df["TENDENCIJA"]

# output vector
y = df['KONCENTRACIJA'].as_matrix()
del df['KONCENTRACIJA']

# binarize id columns
id_cols = ["vrstaBiljke", "ID_VRSTE", "ID_LOKACIJE"]
id_columns = df[id_cols].as_matrix()

for col in id_cols:
    del df[col]

id_columns_binarized = binarize(id_columns[:, 0])

for i in range(1, np.size(id_cols)):
    id_columns_binarized = np.column_stack((id_columns_binarized, binarize(id_columns[:, i])))

# transform to suitable representation
X = df.as_matrix()
X = np.column_stack((X, id_columns_binarized))
# y = binarize(y)

# clear from n/a's
clear_rows = ~np.isnan(X).any(axis=1)
X = X[clear_rows]
y = y[clear_rows]

# create polynomial features
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)
print('Polynomial shape(X) = ', np.shape(X))

# cut off features with low variance
sel = VarianceThreshold(threshold=0.1)
X = sel.fit_transform(X)
print('Without low var features shape(X) = ', np.shape(X))

y[y == 3] = 2

# get the minimum sample
under_sample = sys.maxsize
for i in range(min(y), max(y) + 1):
    temp_sum = sum(y == i)
    if temp_sum < under_sample:
        under_sample = temp_sum

for i in range(min(y), max(y) + 1):
    print('class ', i, ' samples = ', sum(y == i))

# unders-ample each of the oversampled classes
X_leftover = []
y_leftover = []
for i in range(min(y), max(y) + 1):
    X_current_class = X[y == i]
    y_current_class = y[y == i]

    X = X[y != i]
    y = y[y != i]

    # taking full 1-th class, and 4 times lesser 0-th class
    if i == 1:
        k = 20
    else:
        if i == 0:
            k = 1 / 4

    # under-sample the current class
    X = np.row_stack((X, X_current_class[0:k * under_sample, :]))
    y = np.concatenate((y, y_current_class[0:k * under_sample]), axis=0)

    # save the leftover training examples
    if not len(X_leftover) and not len(y_leftover):
        X_leftover = X_current_class[k * under_sample:, :]
        y_leftover = y_current_class[k * under_sample:]
    else:
        X_leftover = np.row_stack((X_leftover, X_current_class[k * under_sample:, :]))
        y_leftover = np.concatenate((y_leftover, y_current_class[k * under_sample:]), axis=0)

    X, y = shuffle(X, y, random_state=0)
    X_leftover, y_leftover = shuffle(X_leftover, y_leftover, random_state=0)

for i in range(min(y), max(y) + 1):
    print('sample size of class ', i, ' =', sum(y == i))

print('X_leftover size = ', np.shape(X_leftover))
print('y_leftover size = ', np.shape(y_leftover))

for i in range(min(y), max(y) + 1):
    print('class ', i, ' samples after undersampling = ', sum(y == i))

# scale the features so that we're left features who got 0 mean and variance 1
std = StandardScaler()
X = std.fit_transform(X)

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# scale the test set with training set mean & variance
# and add the leftover ones for testing
X_leftover = (X_leftover - std.mean_) / std.var_
y_test = np.concatenate((y_leftover, y_test), axis=0)
X_test = np.row_stack((X_leftover, X_test))
X_test, y_test = shuffle(X_test, y_test, random_state=0)

for i in range(min(y), max(y) + 1):
    print('class ', i, ' training samples = ', sum(y_train == i))
for i in range(min(y), max(y) + 1):
    print('class ', i, ' test samples = ', sum(y_test == i))


# blend try
"""
best = 0
for i in range(0, 100):
    current_score = blender(X_train, y_train, X_test, y_test)
    if current_score > best:
        best = current_score
"""

# RDF prediction model
clf = RandomForestClassifier(n_estimators=5000, class_weight='balanced')
clf.fit(X_train, y_train)

# fscore plot
fscore(clf, X_test, y_test)

# TODO: Plot learning curves

# TODO: Exploratory data analysis, manual feature creation(hate this part)
