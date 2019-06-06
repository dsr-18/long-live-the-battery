from os.path import join

# Feature dimensions
steps = 1000  # number of steps in detail level features, e.g. Qdlin and Tdlin
input_dim = 1  # dimensions of detail level features, e.g. Qdlin and Tdlin

# Feature names - use these for matching features in dataset with model inputs
internal_resistance_name = 'IR'
discharge_time_name = 'Discharge_time'
tdlin_name = 'Tdlin'
qdlin_name = 'Qdlin'
remaining_cycles_name = 'Remaining_cycles'

# File paths
train_test_split = "train_test_split.pkl"  # file location for train/test split definition
processed_data = join("data", "processed_data.pkl")  # file location for processed data
datasets_dir = join("data", "tfrecords")  # base directory to write tfrecord files in
tensorboard_dir = "Graph"  # base directory to write tensorboard logs in
trained_model_dir = "./"  # base directory to save trained model in
train_set = join("data", "tfrecords", "train", "*tfrecord")  # regexp files for the training set
test_set = join("data", "tfrecords", "test", "*tfrecord")  # regexp for the test set
secondary_test_set = join("data", "tfrecords", "secondary_test", "*tfrecord")  # regexp for the secondary test set

