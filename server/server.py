import tensorflow as tf
import numpy as np
import flask
from flask import request
import json

import trainer.data_pipeline as dp
import trainer.constants as cst


app = flask.Flask(__name__)
model = None


def load_model():
    global model  # bc YOLO
    model_dir = "saved_models/20190615-171709"      # TODO replace with Docker env dir
    model = tf.keras.experimental.load_from_saved_model(model_dir)


@app.route('/predict', methods=['POST'])
def predict():
    res = { 'success': False }

    if flask.request.method == 'POST':
        # read payload json
        req = request.get_json()

        cycles = { 'Qdlin': np.array(json.loads(req['Qdlin'])),
                   'Tdlin': np.array(json.loads(req['Tdlin'])),
                   'IR': np.array(json.loads(req['IR'])),
                   'Discharge_time': np.array(json.loads(req['Discharge_time'])),
                   'QD': np.array(json.loads(req['QD']))
                }

        predictions = model.predict(cycles)

        print(type(predictions))
        print(predictions)

        res['predictions'] = json.dumps(predictions.tolist())
        res['success'] = True


    return flask.jsonify(res)


if __name__ == '__main__':
    print('--> Loading Keras Model and starting server')
    load_model()
    app.run()


