import pickle
import os

import tensorflow as tf
from tensorflow.train import FloatList, Int64List, Feature, Features, Example

import trainer.constants as cst


def get_cycle_example(cell_value, summary_idx, cycle_idx):
    """
    Define the columns that should be written to tfrecords and converts the processed data
    to "Example" objects. Every Example contains data from one charging cycle.
    """
    cycle_example = Example(
        features=Features(
            feature={
                cst.INTERNAL_RESISTANCE_NAME:
                    Feature(float_list=FloatList(value=[cell_value["summary"][cst.INTERNAL_RESISTANCE_NAME][summary_idx]])),
                cst.QD_NAME:
                    Feature(float_list=FloatList(value=[cell_value["summary"][cst.QD_NAME][summary_idx]])),
                cst.REMAINING_CYCLES_NAME:
                    Feature(float_list=FloatList(value=[cell_value["summary"][cst.REMAINING_CYCLES_NAME][summary_idx]])),
                cst.DISCHARGE_TIME_NAME:
                    Feature(float_list=FloatList(value=[cell_value["summary"][cst.DISCHARGE_TIME_NAME][summary_idx]])),
                cst.QDLIN_NAME:
                    Feature(float_list=FloatList(value=cell_value["cycles"][cycle_idx][cst.QDLIN_NAME])),
                cst.TDLIN_NAME:
                    Feature(float_list=FloatList(value=cell_value["cycles"][cycle_idx][cst.TDLIN_NAME]))
            }
        )
    )
    return cycle_example


def write_to_tfrecords(batteries, data_dir, train_test_split=None):
    """
    Takes processed battery data in dict format as input and writes a set of tfrecords files to disk.

    To load the preprocessed battery data that was used to train the model, store the preprocessed data as 
    "processed_data.pkl" in the data directory and then use "load_processed_battery_data()" to load it.

    A train/test split can be passed as a dictionary with the names of the splits (e.g. "Train") as keys
    and lists of cell names (e.g. ["b1c3", "b1c4"]) as values. This will create subdirectories for each
    split.

    For more info on TFRecords and Examples see 'Hands-on Machine Learning with
    Scikit-Learn, Keras & TensorFlow', pp.416 (2nd edition, early release)
    """
    # Create base directory for tfrecords
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if train_test_split is None:
        # Write all cells into one directory
        for cell_name, cell_data in batteries.items():
            write_single_cell(cell_name, cell_data, data_dir)
    else:
        # For each split set a new working directory in /Data/tfrecords
        # and write files there
        for split_name, split_indexes in train_test_split.items():
            split_data_dir = os.path.join(data_dir, split_name)
            # create directories
            if not os.path.exists(split_data_dir):
                os.mkdir(split_data_dir)
            split_batteries = {idx: batteries[idx] for idx in split_indexes}
            for cell_name, cell_data in split_batteries.items():
                write_single_cell(cell_name, cell_data, split_data_dir)


def write_single_cell(cell_name, cell_data, data_dir):
    """
    Takes data for one cell and writes it to a tfrecords file with the naming convention
    "b1c0.tfrecord". The SerializeToString() method creates binary data out of the
    Example objects that can be read natively in TensorFlow.
    """
    filename = os.path.join(data_dir, cell_name + ".tfrecord")
    with tf.io.TFRecordWriter(str(filename)) as f:
        for summary_idx, cycle_idx in enumerate(cell_data["cycles"].keys()):
            cycle_to_write = get_cycle_example(cell_data, summary_idx, cycle_idx)
            f.write(cycle_to_write.SerializeToString())
    print("Created %s.tfrecords file." % cell_name)


def parse_features(example_proto):
    """
    The parse_features function takes an example and converts it from binary/message format
    into a more readable format. To be able to feed the dataset directly into a
    Tensorflow model later on, we split the data into examples and targets (i.e. X and y).

    The feature_description defines the schema/specifications to read from TFRecords.
    This could also be done by declaring feature columns and parsing the schema
    with tensorflow.feature_columns.make_parse_example_spec().
    """
    feature_description = {
        cst.INTERNAL_RESISTANCE_NAME: tf.io.FixedLenFeature([1, ], tf.float32),
        cst.QD_NAME: tf.io.FixedLenFeature([1, ], tf.float32),
        cst.DISCHARGE_TIME_NAME: tf.io.FixedLenFeature([1, ], tf.float32),
        cst.REMAINING_CYCLES_NAME: tf.io.FixedLenFeature([], tf.float32),
        cst.TDLIN_NAME: tf.io.FixedLenFeature([cst.STEPS, cst.INPUT_DIM], tf.float32),
        cst.QDLIN_NAME: tf.io.FixedLenFeature([cst.STEPS, cst.INPUT_DIM], tf.float32)
    }
    examples = tf.io.parse_single_example(example_proto, feature_description)
    targets = examples.pop(cst.REMAINING_CYCLES_NAME)
    return examples, targets


def get_flatten_windows(window_size):
    def flatten_windows(features, target):
        """
        The .window() function created datasets within the dataset which we need to flatten.
        This function turns every nested dataset into a tensor and stores them in a dict
        that can be read directly by the model.
        """
        # Select all rows for each feature
        qdlin = features[cst.QDLIN_NAME].batch(window_size)
        tdlin = features[cst.TDLIN_NAME].batch(window_size)
        ir = features[cst.INTERNAL_RESISTANCE_NAME].batch(window_size)
        dc_time = features[cst.DISCHARGE_TIME_NAME].batch(window_size)
        qd = features[cst.QD_NAME].batch(window_size)
        # the names in this dict have to match the names of the Input objects in
        # our final model
        features_flat = {
            cst.QDLIN_NAME: qdlin,
            cst.TDLIN_NAME: tdlin,
            cst.INTERNAL_RESISTANCE_NAME: ir,
            cst.DISCHARGE_TIME_NAME: dc_time,
            cst.QD_NAME: qd
        }
        # For every window we want to have one target/label
        # so we only get the last row by skipping all but one row
        target_flat = target.skip(window_size-1)
        return tf.data.Dataset.zip((features_flat, target_flat))
    return flatten_windows


def normalize_feature(feature_name, window_data):
    return tf.math.divide(tf.math.subtract(window_data[feature_name],
                                           tf.math.reduce_min(window_data[feature_name])),
                          tf.math.subtract(tf.math.reduce_max(window_data[feature_name]),
                                           tf.math.reduce_min(window_data[feature_name])))
    
    
def scale_features(window_data, window_target):
    # Normalize values for one window: (data - min) / (max - min).
    window_data[cst.TDLIN_NAME] = normalize_feature(cst.TDLIN_NAME, window_data)
    window_data[cst.QDLIN_NAME] = normalize_feature(cst.QDLIN_NAME, window_data)
    window_data[cst.INTERNAL_RESISTANCE_NAME] = normalize_feature(cst.INTERNAL_RESISTANCE_NAME, window_data)
    window_data[cst.DISCHARGE_TIME_NAME] = normalize_feature(cst.DISCHARGE_TIME_NAME, window_data)
    
    # Scale scalar features.
    window_data[cst.QD_NAME] = tf.math.divide(window_data[cst.QD_NAME], cst.QD_SCALE_FACTOR)
    window_target = tf.math.divide(window_target, cst.REMAINING_CYCLES_SCALE_FACTOR)
    
    return window_data, window_target


def get_create_cell_dataset_from_tfrecords(window_size, shift, stride, drop_remainder, batch_size):
    def create_cell_dataset_from_tfrecords(file):
        """
        The read_tfrecords() function reads a file, skipping the first row which in our case
        is 0/NaN most of the time. It then loops over each example/row in the dataset and
        calls the parse_feature function. Then it batches the dataset, so it always feeds
        multiple examples at the same time, then shuffles the batches. It is important
        that we batch before shuffling, so the examples within the batches stay in order.
        """
        dataset = tf.data.TFRecordDataset(file)
        dataset = dataset.map(parse_features)
        dataset = dataset.window(size=window_size, shift=shift, stride=stride, drop_remainder=drop_remainder)
        dataset = dataset.flat_map(get_flatten_windows(window_size))
        dataset = dataset.map(scale_features)
        dataset = dataset.batch(batch_size)
        return dataset
    return create_cell_dataset_from_tfrecords


def create_dataset(data_dir, window_size, shift, stride, batch_size, 
                   cycle_length=4, num_parallel_calls=4,
                   drop_remainder=True, shuffle=True,
                   shuffle_buffer=500, repeat=True):
    """
    Creates a dataset from .tfrecord files in the data directory. Expects a regular expression
    to capture multiple files (e.g. "data/tfrecords/train/*tfrecord").
    The dataset will augment the original data by creating windows of loading cycles.

    Notes about the interleave() method:
    interleave() will create a dataset that pulls 4 (=cycle_length) file paths from the
    filepath_dataset and for each one calls the function "read_tfrecords()". It will then
    cycle through these 4 datasets, reading one line at a time from each until all datasets
    are out of items. Then it gets the next 4 file paths from the filepath_dataset and
    interleaves them the same way, and so on until it runs out of file paths.
    Even with parallel calls specified, data within batches is sequential.
    """
    filepath_dataset = tf.data.Dataset.list_files(data_dir)
    assembled_dataset = filepath_dataset.interleave(get_create_cell_dataset_from_tfrecords(window_size, shift, stride,
                                                                                           drop_remainder,
                                                                                           batch_size),
                                                    cycle_length=cycle_length,
                                                    num_parallel_calls=num_parallel_calls)
    if shuffle:
        assembled_dataset = assembled_dataset.shuffle(shuffle_buffer)
    if repeat:
        assembled_dataset = assembled_dataset.repeat()
    return assembled_dataset


# dev method
def load_train_test_split():
    """
    Loads a train_test_split dict that divides all cell names into three lists,
    recreating the splits from the original paper.
    This can be passed directly to "write_to_tfrecords()" as an argument.
    """
    return pickle.load(open(cst.TRAIN_TEST_SPLIT, "rb"))


# dev method
def load_processed_battery_data():
    return pickle.load(open(cst.PROCESSED_DATA, "rb"))


if __name__ == "__main__":
    print("Writing datasets with train/test split from original paper and preprocessed data.")
    print("Loading split...")
    split = load_train_test_split()
    print("Loading battery data...")
    battery_data = load_processed_battery_data()
    print("Start writing to disk...")
    write_to_tfrecords(battery_data, cst.DATASETS_DIR, train_test_split=split)
    print("Done.")
