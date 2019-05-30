import os
import glob
from tensorflow.train import FloatList, Int64List, Feature, Features, Example
import tensorflow as tf


def get_cycle_example(cell, idx):
    """
    Define the columns that should be written to tfrecords and format the raw data
    which comes in a dictionary-like format from pickled battery files.
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


def write_to_tfrecords(batteries, data_dir="Data/tfrecords/"):
    """
    Takes pickled battery data as input. Use "load_batches_to_dict()" from
    the "rebuilding_features.py" module to load the battery data.

    1. "get_cycle_features()" fetches all features and targets from
    the battery data and converts to "Example" objects. Every Example contains
    data from one charging cycle.

    2. For each cell create a tfrecord file with the naming convention "b1c0.tfrecord".
    The SerializeToString() method creates binary data out of the Example objects that can
    be read natively in TensorFlow.

    For more info on TFRecords and Examples see 'Hands-on Machine Learning with
    Scikit-Learn, Keras & TensorFlow', pp.416 (2nd edition, early release)
    """
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for cell_name, cell_data in batteries.items():
        filename = os.path.join(data_dir + cell_name + ".tfrecord")
        with tf.io.TFRecordWriter(filename) as f:
            for cycle_idx in cell_data["summary"]["cycle"]:
                cycle_to_write = get_cycle_example(cell_data, int(cycle_idx))
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
        'Remaining_cycles': tf.io.FixedLenFeature([], tf.int64),
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


def get_create_cell_dataset_from_tfrecords(window_size, shift, stride, drop_remainder, batch_size, shuffle):
    def create_cell_dataset_from_tfrecords(file):
        """
        The read_tfrecords() function reads a file, skipping the first row which in our case
        is 0/NaN most of the time. It then loops over each example/row in the dataset and
        calls the parse_feature function. Then it batches the dataset, so it always feeds
        multiple examples at the same time, then shuffles the batches. It is important
        that we batch before shuffling, so the examples within the batches stay in order.
        """
        dataset = tf.data.TFRecordDataset(file).skip(1)  # .skip() should be removed when we have clean data
        dataset = dataset.map(parse_features)
        dataset = dataset.window(size=window_size, shift=shift, stride=stride, drop_remainder=drop_remainder)
        dataset = dataset.flat_map(get_flatten_windows(window_size))
        dataset = dataset.batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(1000)
        return dataset
    return create_cell_dataset_from_tfrecords


def create_dataset(data_dir="Data/tfrecords/", cycle_length=4, num_parallel_calls=4,
                   window_size=5, shift=1, stride=1, drop_remainder=True, batch_size=10, shuffle=True):
    """
    The interleave() method will create a dataset that pulls 4 (=cycle_length) file paths from the
    filepath_dataset and for each one calls the function "read_tfrecords()". It will then
    cycle through these 4 datasets, reading one line at a time from each until all datasets
    are out of items. Then it gets the next 4 file paths from the filepath_dataset and
    interleaves them the same way, and so on until it runs out of file paths.
    Note: Even with parallel calls specified, data within batches is sequential.
    """
    filepaths = glob.glob(os.path.join(data_dir + "*.tfrecord"))
    filepath_dataset = tf.data.Dataset.list_files(filepaths)
    assembled_dataset = filepath_dataset.interleave(get_create_cell_dataset_from_tfrecords(window_size, shift, stride,
                                                                                           drop_remainder,
                                                                                           batch_size, shuffle),
                                                    cycle_length=cycle_length,
                                                    num_parallel_calls=num_parallel_calls)
    assembled_dataset = assembled_dataset.shuffle(1000)
    return assembled_dataset
