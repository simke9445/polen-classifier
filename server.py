import pickle
import numpy as np

from flask import Flask, request, jsonify, json
from binarize_single import binarize

# load the classifer
with open('classifier.pickle', 'rb') as handle:
    clf = pickle.load(handle)

# load the polynomial feauture transformer
with open('polynomial-transformer.pickle', 'rb') as handle:
    poly = pickle.load(handle)

# load the standard scaler
with open('scaler.pickle', 'rb') as handle:
    std = pickle.load(handle)

# load the variance feature selector
with open('variance-threshholder.pickle', 'rb') as handle:
    sel = pickle.load(handle)

# set the global app to main
app = Flask(__name__)


# main page, index
@app.route('/')
def hello_world():
    return "Index page, nothing here"


# route for getting the prediction out of a post request,
# which passes a JSON object as a parameter containing
# necessary values for making the prediction
@app.route("/data/", methods=['POST', 'GET'])
def handle_data():
    # load the json passed params
    json_object = json.loads(request.get_json())
    print(json_object)

    # stack up the initial test vector
    vector = np.column_stack((json_object['dan'],
                              json_object['mesec'],
                              json_object['godina'],
                              json_object['alergenost'],
                              json_object['nadmorskaVisina'],
                              json_object['nadmorskaSirina']))

    # binarize id columns
    id_cols = ["vrstaBiljke", "ID_VRSTE", "ID_LOKACIJE"]
    id_columns = np.column_stack((json_object['vrstaBiljke'],
                                  json_object['ID_VRSTE'],
                                  json_object['ID_LOKACIJE']))

    # testing binarized id columns
    binarized_vrstaBiljke = binarize(int(id_columns[:, 0]), 0, 2)
    binarized_id_lokacije = binarize(int(id_columns[:, 1]), 1, 18)
    binarized_id_vrste = binarize(int(id_columns[:, 2]), 1, 25)

    print('vector initial size: ', vector.shape)
    print('id_columns initial size: ', id_columns.shape)
    print('vrsta biljke: ', binarized_vrstaBiljke, ' real value = ', id_columns[0, 0])
    print('id_lokacije: ', binarized_id_lokacije, ' real value = ', id_columns[0, 1])
    print('id_vrste: ', binarized_id_vrste, ' real value = ', id_columns[0, 2])

    # add them to testing vector
    vector = np.column_stack((vector,
                              binarized_vrstaBiljke,
                              binarized_id_lokacije,
                              binarized_id_vrste))

    # testing the shapes before & after the transformations
    print('vector after size: ', vector.shape)
    print(vector.shape)

    # apply polynomial features
    vector = poly.transform(vector)
    print(vector.shape)

    # apply variance feature selection
    vector = sel.transform(vector)
    print(vector.shape)

    # apply standard scaling(zero mean & unit variance)
    vector = std.transform(vector)

    # prediction
    print(clf.predict_proba(vector))

    # predicting the test vector and converting it to json which
    # will be returned to request
    prediction = clf.predict(vector)
    print(prediction)
    prediction_json = {'class': float(prediction[0])}

    print(prediction_json)

    return jsonify(prediction_json)


if __name__ == '__main__':
    app.run(debug=True)
