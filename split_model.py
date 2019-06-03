from tensorflow.keras.layers import concatenate, LSTM, Conv1D, Flatten, TimeDistributed, Input, Dense
from tensorflow.keras.models import Model

# TODO use the args
def create_keras_model(args):
  """Trains and evaluates the Keras model.

  Uses the Keras model defined in model.py and trains on data loaded and
  preprocessed in data_pipeline.py. Saves the trained model in TensorFlow SavedModel
  format to the path defined in part by the --job-dir argument.

  Args:
    args: dictionary of arguments - see get_args() for details
  """
  window_size = 5
  steps = 1000
  input_dim = 1

  # define Inputs
  qdlin_in = Input(shape=(window_size, steps, input_dim), name='Qdlin')
  tdlin_in = Input(shape=(window_size, steps, input_dim), name='Tdlin')
  ir_in = Input(shape=(window_size, input_dim), name='IR')

  # combine all data from detail level
  detail_concat = concatenate([qdlin_in, tdlin_in], axis=3, name='detail_concat')

  # define CNN
  cnn_out = TimeDistributed(Conv1D(filters=3, kernel_size=5, activation='relu'), name='convolution')(detail_concat)
  cnn_flat = TimeDistributed(Flatten(), name='convolution_flat')(cnn_out)

  # combine CNN output with all data from summary level
  all_concat = concatenate([cnn_flat, ir_in], axis=2, name='all_concat')

  # define LSTM
  lstm_out = LSTM(20, activation='relu', name='recurrent')(all_concat)
  main_output = Dense(1, name='output')(lstm_out)

  model = Model(inputs=[qdlin_in, tdlin_in, ir_in], outputs=[main_output])
  model.compile(loss='mean_squared_error', optimizer='adam')

  return model