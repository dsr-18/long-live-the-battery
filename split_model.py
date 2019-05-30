from tensorflow.keras.layers import concatenate, LSTM, Dense, Conv1D, Flatten, TimeDistributed, Activation, Input, Dense
from tensorflow.keras.models import Model
import data_pipeline as dp

# Test dataset compatibility with a CNN + LSTM model layout

steps = 1000
input_dim = 1
window_size = 5

main_input = Input(shape=(window_size,
                          steps,
                          input_dim),
                   name="Qdlin")

# define CNN model
cnn_out = TimeDistributed(Conv1D(filters=1, kernel_size=3, activation='relu'))(main_input)
cnn_flat = TimeDistributed(Flatten())(cnn_out)
cnn_dense = TimeDistributed(Dense(1))(cnn_flat)

# define LSTM model
lstm_out = LSTM(50, activation='relu')(cnn_dense)
main_output = Dense(1)(lstm_out)

model = Model(inputs=[main_input], outputs=[main_output])
model.compile(loss="mean_squared_error", optimizer='adam')


dataset = dp.create_dataset()

history = model.fit(dataset, epochs=1, steps_per_epoch=4)
print(history)