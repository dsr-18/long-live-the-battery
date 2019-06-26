import json
from random import randint
import os

import flask
import numpy as np
import plotly
import plotly.graph_objs as go
import tensorflow as tf
from flask import Flask, render_template, request
from plot import plot_single_prediction
from constants import NUM_SAMPLES, MODEL_DIR, SAMPLES_DIR
from clippy import Clippy, clipped_relu

app = Flask(__name__)


def load_model():
    global model  # bc YOLO
    model = tf.keras.experimental.load_from_saved_model(MODEL_DIR, custom_objects={'clippy': Clippy(clipped_relu)})


def make_prediction(cycle_data, response):
    cycles = { 'Qdlin': np.array(json.loads(cycle_data['Qdlin'])),
                'Tdlin': np.array(json.loads(cycle_data['Tdlin'])),
                'IR': np.array(json.loads(cycle_data['IR'])),
                'Discharge_time': np.array(json.loads(cycle_data['Discharge_time'])),
                'QD': np.array(json.loads(cycle_data['QD']))
            }

    predictions = model.predict(cycles)

    print("Returning predictions:")
    print(type(predictions))
    print(predictions)

    response['predictions'] = json.dumps(predictions.tolist())
    response['success'] = True
    
    return flask.jsonify(response)


def make_plot(predictions):
    predictions = np.array(predictions)
    # The prediction endpoint can handle batches of battery data per request,
    # but for now we visualize only the first data example.
    first_pred = predictions[0]
    window_size = model.input_shape[0][1]
    scaling_factors_dict = {"Remaining_cycles": 2159.0}
    mean_cycle_life = 674  # calculated from training set
    figure = plot_single_prediction(first_pred,
                                  window_size,
                                  scaling_factors_dict,
                                  mean_cycle_life)
    gaphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    return gaphJSON


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title="Home")


@app.route('/predict', methods=['POST'])
def predict():
    res = { 'success': False }

    if flask.request.method == 'POST':
        # read payload json
        if len(request.files) > 0:
            print("form upload")
            parsed_data = request.files["jsonInput"].read().decode('utf8')
        elif request.get_json() is None:
            print("example upload")
            parsed_data = request.form["jsonInput"]
            parsed_data = parsed_data.replace("'", '"')
        else:
            print("curl upload")
            json_data = request.get_json()
            return make_prediction
        
    json_data = json.loads(parsed_data)
    predictions_response = make_prediction(json_data, res)
    predictions = json.loads(predictions_response.json["predictions"])
    plot = make_plot(predictions)
    
    return render_template("results.html", title="Results", plot=plot)


@app.route('/example')
def example():
    if request.args:
        # on request for a sample file, pick a random json file with prepared 
        # battery data and return its content
        rand = randint(1,NUM_SAMPLES)
        filename = "sample_input_{}.json".format(rand)
        with open(os.path.join(SAMPLES_DIR,"{}".format(filename)), "r") as json_file:
            json_data = json.load(json_file)
    else:
        # on siteload return an empty object so the javascript disables 
        # buttons and does not show a filename in the input
        filename = ""
        json_data = None
    return render_template("example.html", title="Samples", filename=filename, data=json_data)

    
if __name__ == "__main__":
    print('--> Loading Keras Model and starting server')
    model = None
    load_model()        
    app.run(host="0.0.0.0")
