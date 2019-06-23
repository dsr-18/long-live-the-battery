import json

import flask
import numpy as np
import plotly
import plotly.graph_objs as go
import tensorflow as tf
from flask import Flask, render_template, request


app = Flask(__name__)

def load_model():
    global model  # bc YOLO
    model_dir = "saved_model/"
    model = tf.keras.experimental.load_from_saved_model(model_dir)


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
    x = np.sort(predictions[:,0])
    y = predictions[:,1]
    data = [go.Scatter(x=x, y=y)]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


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
            print("Upload via form")
            parsed_data = request.files["jsonInput"].read().decode('utf8')
            json_data = json.loads(parsed_data)
            predictions_response = make_prediction(json_data, res)
            predictions = json.loads(predictions_response.json["predictions"])
            plot = make_plot(predictions)
            return render_template("results.html", title="Results", prediction=predictions, plot=plot)
        else:
            print("Upload via curl")
            json_data = request.get_json()
            return make_prediction(json_data, res)
        
if __name__ == "__main__":
    print('--> Loading Keras Model and starting server')
    model = None
    load_model()        
    app.run(host="0.0.0.0")
