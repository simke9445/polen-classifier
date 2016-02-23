import json

import pandas as pd
import requests

"""
Script for testing the server/client communication
"""

# preprocess the input data
df = pd.read_csv("polen_data.csv")

# shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# remove unnecessary columns
del df["Unnamed: 0"], df["TENDENCIJA"]

# output vector
y = df['KONCENTRACIJA'].as_matrix()

del df['KONCENTRACIJA']

print(df.columns.values.tolist())

print('min[id_vrste] = ', min(df['ID_VRSTE']))
print('min[id_lokacije] = ', min(df['ID_LOKACIJE']))
print('min[vrstaBiljke] = ', min(df['vrstaBiljke']))

print('max[id_vrste] = ', max(df['ID_VRSTE']))
print('min[id_lokacije] = ', max(df['ID_LOKACIJE']))
print('max[vrstaBiljke] = ', max(df['vrstaBiljke']))

# init request url
url = 'http://127.0.0.1:5000/data/'

for index in range(0, df.shape[0]):
    # init payload
    payload = df.iloc[index].to_json()
    print(payload)
    headers = {'content-type': 'application/json'}

    response = requests.post(url, data=json.dumps(payload), headers=headers)

    print(response.text)
