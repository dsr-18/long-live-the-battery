from tensorflow.keras.layers import concatenate, LSTM, Conv1D, Flatten, TimeDistributed, Input, Dense
from tensorflow.keras.models import Model


def create_keras_model(window_size, loss, optimizer):
  """Creates the Keras model.
  
  Args:
    window_size: [...]
  """
    steps = 1000
    input_dim = 1

    # define Inputs
    qdlin_in = Input(shape=(window_size, steps, input_dim), name='Qdlin')
    tdlin_in = Input(shape=(window_size, steps, input_dim), name='Tdlin')
    ir_in = Input(shape=(window_size, input_dim), name='IR')
    dt_in = Input(shape=(window_size, input_dim), name='Discharge_time')

    # combine all data from detail level
    detail_concat = concatenate([qdlin_in, tdlin_in], axis=3, name='detail_concat')

    # define CNN
    cnn_out = TimeDistributed(Conv1D(filters=3, kernel_size=5, activation='relu'), name='convolution')(detail_concat)
    cnn_flat = TimeDistributed(Flatten(), name='convolution_flat')(cnn_out)

    # combine CNN output with all data from summary level
    all_concat = concatenate([cnn_flat, ir_in, dt_in], axis=2, name='all_concat')

    # define LSTM
    lstm_out = LSTM(20, activation='relu', name='recurrent')(all_concat)
    main_output = Dense(1, name='output')(lstm_out)

    model = Model(inputs=[qdlin_in, tdlin_in, ir_in, dt_in], outputs=[main_output])
    model.compile(loss=loss, optimizer=optimizer)

    return model
