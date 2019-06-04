import os
import glob
from tensorflow.train import FloatList, Int64List, Feature, Features, Example
import tensorflow as tf
import pickle

TFR_DIR = "data/tfrecords/"

def get_cycle_example(cell, idx):
    """
    Define the columns that should be written to tfrecords and converts the raw data
    to "Example" objects. Every Example contains data from one charging cycle.
    """
    cycle_example = Example(
        features=Features(
            feature={
                "IR": Feature(float_list=FloatList(value=[cell["summary"]["IR"][idx]])),
                "Qdlin": Feature(float_list=FloatList(value=cell["cycles"][str(idx)]["Qdlin"])),
                "Tdlin": Feature(float_list=FloatList(value=cell["cycles"][str(idx)]["Tdlin"])),
                "Remaining_cycles": Feature(int64_list=Int64List(value=[int(cell["cycle_life"]-idx)]))
            }
        )
    )
    return cycle_example


def get_preprocessed_cycle_example(cell, idx):
    """
    Same as above, but with the preprocessed data
    """
    cycle_example = Example(
        features=Features(
            feature={
                "IR":
                    Feature(float_list=FloatList(value=[cell["summary"]["IR"][idx]])),
                "Remaining_cycles":
                    Feature(int64_list=Int64List(value=[cell["summary"]["remaining_cycle_life"][idx]])),
                "Discharge_time":
                    Feature(float_list=FloatList(value=[cell["summary"]["high_current_discharging_time"][idx]])),
                "Qdlin":
                    Feature(float_list=FloatList(value=cell["cycles"][str(idx)]["Qd_resample"])),
                "Tdlin":
                    Feature(float_list=FloatList(value=cell["cycles"][str(idx)]["T_resample"]))
            }
        )
    )
    return cycle_example


def write_to_tfrecords(batteries, data_dir="Data/tfrecords/", preprocessed=True, train_test_split=None):
    """
    Takes battery data in dict format as input and writes a set of tfrecords files to disk.

    To load the preprocessed battery data that was used to train the model, you can use
    "load_preprocessed_data_to_dict()" from the "data_preprocessing.py" module.
    To load unprocessed battery data you can use "load_batches_to_dict()" from the
    "rebuilding_features.py" module and set "preprocessed" to False.

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
            write_single_cell(cell_name, cell_data, data_dir, preprocessed)
    else:
        # For each split set a new working directory in /Data/tfrecords
        # and write files there
        for split_name, split_indexes in train_test_split.items():
            split_data_dir = os.path.join(data_dir + split_name + "/")
            # create directories
            if not os.path.exists(split_data_dir):
                os.mkdir(split_data_dir)
            split_batteries = {idx: batteries[idx] for idx in split_indexes}
            for cell_name, cell_data in split_batteries.items():
                write_single_cell(cell_name, cell_data, split_data_dir, preprocessed)


def write_single_cell(cell_name, cell_data, data_dir, preprocessed):
    """
    Takes data for one cell and writes it to a tfrecords file with the naming convention
    "b1c0.tfrecord". The SerializeToString() method creates binary data out of the
    Example objects that can be read natively in TensorFlow.
    """
    filename = os.path.join(data_dir + cell_name + ".tfrecord")
    with tf.io.TFRecordWriter(filename) as f:
        for cycle_idx in cell_data["summary"]["cycle"]:
            if preprocessed:
                cycle_to_write = get_preprocessed_cycle_example(cell_data, int(cycle_idx)-1)
            else:
                cycle_to_write = get_cycle_example(cell_data, int(cycle_idx)-1)
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
        'IR': tf.io.FixedLenFeature([1, ], tf.float32),
        'Tdlin': tf.io.FixedLenFeature([1000, 1], tf.float32),
        'Qdlin': tf.io.FixedLenFeature([1000, 1], tf.float32),
        'Remaining_cycles': tf.io.FixedLenFeature([], tf.int64)
    }
    examples = tf.io.parse_single_example(example_proto, feature_description)
    targets = examples.pop("Remaining_cycles")
    return examples, targets


def parse_preprocessed_features(example_proto):
    """
    Same as above, but with preprocessed features.
    """
    feature_description = {
        'IR': tf.io.FixedLenFeature([1, ], tf.float32),
        'Discharge_time': tf.io.FixedLenFeature([1, ], tf.float32),
        'Remaining_cycles': tf.io.FixedLenFeature([], tf.int64),
        'Tdlin': tf.io.FixedLenFeature([1000, 1], tf.float32),
        'Qdlin': tf.io.FixedLenFeature([1000, 1], tf.float32)
    }
    examples = tf.io.parse_single_example(example_proto, feature_description)
    targets = examples.pop("Remaining_cycles")
    return examples, targets


def get_flatten_windows(window_size):
    def flatten_windows(features, target):
        """
        Calling .window() on our dataset created a dataset of type "VariantDataset"
        for every feature in our main dataset. We need to flatten
        these VariantDatasets before we can feed everything to a model.
        Because the VariantDataset are modeled after windows, they have
        length=window_size.
        """
        # Select all rows for each feature
        qdlin = features["Qdlin"].batch(window_size)
        tdlin = features["Tdlin"].batch(window_size)
        ir = features["IR"].batch(window_size)
        # the names in this dict have to match the names of the Input objects in
        # our final model
        features_flat = {
            "Qdlin": qdlin,
            "Tdlin": tdlin,
            "IR": ir
        }
        # For every window we want to have one target/label
        # so we only get the last row by skipping all but one row
        target_flat = target.skip(window_size-1)
        return tf.data.Dataset.zip((features_flat, target_flat))
    return flatten_windows


def get_prep_flatten_windows(window_size):
    def prep_flatten_windows(features, target):
        """
        Same as above, but with the preprocessed data.
        """
        # Select all rows for each feature
        qdlin = features["Qdlin"].batch(window_size)
        tdlin = features["Tdlin"].batch(window_size)
        ir = features["IR"].batch(window_size)
        dc_time = features["Discharge_time"].batch(window_size)
        # the names in this dict have to match the names of the Input objects in
        # our final model
        features_flat = {
            "Qdlin": qdlin,
            "Tdlin": tdlin,
            "IR": ir,
            "Discharge_time": dc_time
        }
        # For every window we want to have one target/label
        # so we only get the last row by skipping all but one row
        target_flat = target.skip(window_size-1)
        return tf.data.Dataset.zip((features_flat, target_flat))
    return prep_flatten_windows


def get_create_cell_dataset_from_tfrecords(window_size, shift, stride, drop_remainder, batch_size, shuffle,
                                           preprocessed=True):
    def create_cell_dataset_from_tfrecords(file):
        """
        The read_tfrecords() function reads a file, skipping the first row which in our case
        is 0/NaN most of the time. It then loops over each example/row in the dataset and
        calls the parse_feature function. Then it batches the dataset, so it always feeds
        multiple examples at the same time, then shuffles the batches. It is important
        that we batch before shuffling, so the examples within the batches stay in order.
        """
        if preprocessed:
            dataset = tf.data.TFRecordDataset(file)
            dataset = dataset.map(parse_preprocessed_features)
            dataset = dataset.window(size=window_size, shift=shift, stride=stride, drop_remainder=drop_remainder)
            dataset = dataset.flat_map(get_prep_flatten_windows(window_size))
        else:
            dataset = tf.data.TFRecordDataset(file).skip(1)
            dataset = dataset.map(parse_features)
            dataset = dataset.window(size=window_size, shift=shift, stride=stride, drop_remainder=drop_remainder)
            dataset = dataset.flat_map(get_flatten_windows(window_size))
        dataset = dataset.batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(1000)
        return dataset
    return create_cell_dataset_from_tfrecords


def create_dataset(data_dir="Data/tfrecords/", cycle_length=4, num_parallel_calls=4,
                   window_size=5, shift=1, stride=1, drop_remainder=True, batch_size=10, shuffle=True,
                   preprocessed=True):
    """
    Creates a dataset from all .tfrecord files in the data directory. The dataset will augment the original data by
    creating windows of loading cycles.

    To load unprocessed data, set "preprocessed" to False.

    Notes about the interleave() method:
    interleave() will create a dataset that pulls 4 (=cycle_length) file paths from the
    filepath_dataset and for each one calls the function "read_tfrecords()". It will then
    cycle through these 4 datasets, reading one line at a time from each until all datasets
    are out of items. Then it gets the next 4 file paths from the filepath_dataset and
    interleaves them the same way, and so on until it runs out of file paths.
    Even with parallel calls specified, data within batches is sequential.
    """
    filepaths = glob.glob(os.path.join(data_dir + "*.tfrecord"))
    filepath_dataset = tf.data.Dataset.list_files(filepaths)
    assembled_dataset = filepath_dataset.interleave(get_create_cell_dataset_from_tfrecords(window_size, shift, stride,
                                                                                           drop_remainder,
                                                                                           batch_size, shuffle,
                                                                                           preprocessed),
                                                    cycle_length=cycle_length,
                                                    num_parallel_calls=num_parallel_calls)
    assembled_dataset = assembled_dataset.shuffle(1000)
    return assembled_dataset


def load_train_test_split():
    """
    Loads a train_test_split dict that divides all cell names into three lists,
    recreating the splits from the original paper.
    This can be passed directly to "write_to_tfrecords()" as an argument.
    """
    return pickle.load(open("train_test_split.pkl", "rb"))
