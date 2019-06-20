import tensorflow as tf
import numpy as np
import flask
from flask import request, render_template, redirect
import json

import trainer.data_pipeline as dp
import trainer.constants as cst

from server import app

model = None


def load_model():
    global model  # bc YOLO
    model_dir = "../data/saved_model/"      # TODO replace with Docker env dir
    model = tf.keras.experimental.load_from_saved_model(model_dir)


def make_prediction(cycle_data, res):
    cycles = { 'Qdlin': np.array(json.loads(cycle_data['Qdlin'])),
                'Tdlin': np.array(json.loads(cycle_data['Tdlin'])),
                'IR': np.array(json.loads(cycle_data['IR'])),
                'Discharge_time': np.array(json.loads(cycle_data['Discharge_time'])),
                'QD': np.array(json.loads(cycle_data['QD']))
            }

    predictions = model.predict(cycles)

    print(type(predictions))
    print(predictions)

    res['predictions'] = json.dumps(predictions.tolist())
    res['success'] = True
    
    return flask.jsonify(res)


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title="Home")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    res = { 'success': False }

    if flask.request.method == 'POST':
        # read payload json
        if len(request.files) > 0:
            print("Upload via form")
            parsed_data = request.files["myjson"].read().decode('utf8')
            json_data = json.loads(parsed_data)
            return render_template("results.html", title="Results", data=make_prediction(json_data, res))
        else:
            print("Upload via curl")
            json_data = request.get_json()
            return make_prediction(json_data, res)
        
load_model()
# if __name__ == '__main__':
#     print('--> Loading Keras Model and starting server')
#     app.run()

