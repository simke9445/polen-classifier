import pandas as pd
import scipy as sp
import numpy as np
from binarizer import binarize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

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
y = binarize(y)

# clear from n/a's
clear_rows = ~np.isnan(X).any(axis=1)
X = X[clear_rows]
y = y[clear_rows]

# create polynomial features
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

print('Polynomial shape(X) = ', np.shape(X))

# cut off features with low variance
sel = VarianceThreshold()
X = sel.fit_transform(X)
print('Without low var features shape(X) = ', np.shape(X))

# scale the features so that we're left features who got 0 mean and variance 1
std = StandardScaler()
X = std.fit_transform(X)


# TODO: Split into training, cv, test sets using stratified sampling

# TODO: K-fold cross validation and test sets using stratified sampling

# TODO: Multiclass L2 logistic regression prediction model

# TODO: Plot learning curves

# TODO: Fscore plot

# TODO: Exploratory data analysis, manual feature creation(hate this part)

# TODO: AdaBoost classifier model

# TODO: ExtraTrees clasifier model

# TODO: RDF's classifier model

# TODO: Blending AdaBoost, ExtraTrees, Rdf's for better ROC
