from os.path import join

# Feature dimensions
STEPS = 1000  # number of steps in detail level features, e.g. Qdlin and Tdlin
INPUT_DIM = 1  # dimensions of detail level features, e.g. Qdlin and Tdlin

# Feature names - use these for matching features in dataset with model inputs
INTERNAL_RESISTANCE_NAME = 'IR'
QD_NAME = 'QD'
DISCHARGE_TIME_NAME = 'Discharge_time'
TDLIN_NAME = 'Tdlin'
QDLIN_NAME = 'Qdlin'
VDLIN_NAME = 'Vdlin'
REMAINING_CYCLES_NAME = 'Remaining_cycles'

REMAINING_CYCLES_SCALE_FACTOR = 3000  # Arbitrary number for division of Remaining_cycles, just to make it explicit.

# File paths
TRAIN_TEST_SPLIT = "train_test_split.pkl"  # file location for train/test split definition
PROCESSED_DATA = join("data", "processed_data.pkl")  # file location for processed data
DATASETS_DIR = join("data", "tfrecords")  # base directory to write tfrecord files in
TENSORBOARD_DIR = "Graph"  # base directory to write tensorboard logs in
SAVED_MODELS_DIR_LOCAL = "saved_models" # base directory to save trained model in
BASE_DIR = "./"  # home directory
TRAIN_SET = join("data", "tfrecords", "train", "*tfrecord")  # regexp files for the training set
TEST_SET = join("data", "tfrecords", "test", "*tfrecord")  # regexp for the test set
SECONDARY_TEST_SET = join("data", "tfrecords", "secondary_test", "*tfrecord")  # regexp for the secondary test set
