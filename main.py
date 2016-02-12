import pandas as pd
import scipy as sp
import numpy as np
from binarizer import binarize

## preprocess the input data
df = pd.read_csv("polen_data.csv")

# shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# clear from n/a data
df = df[np.isfinite(df)]

# remove unnecessary columns
del df["Unnamed: 0"], df["TENDENCIJA"]

# output vector
y = df['KONCENTRACIJA'].as_matrix()
binarize(y)
del df['KONCENTRACIJA']

# binarize id columns
id_cols = ["vrstaBiljke", "ID_VRSTE", "ID_LOKACIJE"]
id_columns = df[id_cols].as_matrix()

for col in id_cols:
    del df[col]

id_columns_binarized = binarize(id_columns[:, 0])

for i in range(1, np.size(id_cols)):
    id_columns_binarized = np.column_stack((id_columns_binarized, binarize(id_columns[:, i])))

# input matrix
X = df.as_matrix()
X = np.column_stack((X, id_columns_binarized))


# TODO: Remove unnecessary features with low variance

# TODO: Create polynomial features, degree = 2

# TODO: Scale features using mean/variance scaling

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
