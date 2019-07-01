import csv
import pickle
import os

import tensorflow as tf
from tensorflow.train import FloatList, Feature, Features, Example

import trainer.constants as cst


def get_cycle_example(cell_value, summary_idx, cycle_idx, scaling_factors):
    """
    Define the columns that should be written to tfrecords and converts the raw data
    to "Example" objects. Every Example contains data from one charging cycle.
    The data is scaled (divided) by the corresponding values in "scaling_factors".
    """
    # Summary feature values (scalars --> have to be wrapped in lists)
    ir_value = [cell_value["summary"][cst.INTERNAL_RESISTANCE_NAME][summary_idx]
                / scaling_factors[cst.INTERNAL_RESISTANCE_NAME]]
    qd_value = [cell_value["summary"][cst.QD_NAME][summary_idx]
                / scaling_factors[cst.QD_NAME]]
    rc_value = [cell_value["summary"][cst.REMAINING_CYCLES_NAME][summary_idx]
                / scaling_factors[cst.REMAINING_CYCLES_NAME]]
    dt_value = [cell_value["summary"][cst.DISCHARGE_TIME_NAME][summary_idx]
                / scaling_factors[cst.DISCHARGE_TIME_NAME]]
    cc_value = [float(cycle_idx)
                / scaling_factors[cst.REMAINING_CYCLES_NAME]]  # Same scale --> same scaling factor
    
    # Detail feature values (arrays)
    qdlin_value = cell_value["cycles"][cycle_idx][cst.QDLIN_NAME] / scaling_factors[cst.QDLIN_NAME]
    tdlin_value = cell_value["cycles"][cycle_idx][cst.TDLIN_NAME] / scaling_factors[cst.TDLIN_NAME]
    
    # Wrapping as example
    cycle_example = Example(
        features=Features(
            feature={
                cst.INTERNAL_RESISTANCE_NAME:
                    Feature(float_list=FloatList(value=ir_value)),
                cst.QD_NAME:
                    Feature(float_list=FloatList(value=qd_value)),
                cst.REMAINING_CYCLES_NAME:
                    Feature(float_list=FloatList(value=rc_value)),
                cst.DISCHARGE_TIME_NAME:
                    Feature(float_list=FloatList(value=dt_value)),
                cst.QDLIN_NAME:
                    Feature(float_list=FloatList(value=qdlin_value)),
                cst.TDLIN_NAME:
                    Feature(float_list=FloatList(value=tdlin_value)),
                cst.CURRENT_CYCLE_NAME:
                    Feature(float_list=FloatList(value=cc_value))
            }
        )
    )
    return cycle_example


def write_to_tfrecords(batteries, data_dir, train_test_split=None):
    """
    Takes battery data in dict format as input and writes a set of tfrecords files to disk.

    To load the preprocessed battery data that was used to train the model, use the
    "load_processed_battery_data()" function and pass it as the batteries argument to the
    "Write_to_tfrecords()" function.

    A train/test split can be passed as a dictionary with the names of the splits (e.g. "train") as keys
    and lists of cell names (e.g. ["b1c3", "b1c4"]) as values. This will create subdirectories for each
    split.

    For more info on TFRecords and Examples see 'Hands-on Machine Learning with
    Scikit-Learn, Keras & TensorFlow', pp.416 (2nd edition, early release)
    """
    # Create base directory for tfrecords
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    scaling_factors = calculate_and_save_scaling_factors(batteries, train_test_split, cst.SCALING_FACTORS_DIR)
    
    if train_test_split is None:
        # Write all cells into one directory
        for cell_name, cell_data in batteries.items():
            write_single_cell(cell_name, cell_data, data_dir, scaling_factors)
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
                write_single_cell(cell_name, cell_data, split_data_dir, scaling_factors)


def write_single_cell(cell_name, cell_data, data_dir, scaling_factors):
    """
    Takes data for one cell and writes it to a tfrecords file with the naming convention
    "b1c0.tfrecord". The SerializeToString() method creates binary data out of the
    Example objects that can be read natively in TensorFlow.
    """
    filename = os.path.join(data_dir, cell_name + ".tfrecord")
    with tf.io.TFRecordWriter(str(filename)) as f:
        for summary_idx, cycle_idx in enumerate(cell_data["cycles"].keys()):
            cycle_to_write = get_cycle_example(cell_data, summary_idx, cycle_idx, scaling_factors)
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
        cst.CURRENT_CYCLE_NAME: tf.io.FixedLenFeature([], tf.float32),
        cst.TDLIN_NAME: tf.io.FixedLenFeature([cst.STEPS, cst.INPUT_DIM], tf.float32),
        cst.QDLIN_NAME: tf.io.FixedLenFeature([cst.STEPS, cst.INPUT_DIM], tf.float32)
    }
    examples = tf.io.parse_single_example(example_proto, feature_description)
    
    target_remaining = examples.pop(cst.REMAINING_CYCLES_NAME)
    target_current = examples.pop(cst.CURRENT_CYCLE_NAME)
    targets = tf.stack([target_current, target_remaining], 0)
    
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
        target_flat = target.skip(window_size - 1)
        return tf.data.Dataset.zip((features_flat, target_flat))
    return flatten_windows


def get_create_cell_dataset_from_tfrecords(window_size, shift, stride, drop_remainder):
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

    To load unprocessed data, set "preprocessed" to False.

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
                                                                                           drop_remainder),
                                                    cycle_length=cycle_length,
                                                    num_parallel_calls=num_parallel_calls)
    if shuffle:
        assembled_dataset = assembled_dataset.shuffle(shuffle_buffer)
    
    # The batching has to happen after shuffling the windows, so one batch is not sequential
    assembled_dataset = assembled_dataset.batch(batch_size)
    
    if repeat:
        assembled_dataset = assembled_dataset.repeat()
    return assembled_dataset


def calculate_and_save_scaling_factors(data_dict, train_test_split, csv_dir):
    """Calculates the scaling factors for every feature based on the training set in train_test_split
    and saves the result in a csv file. The factors are used during writing of the tfrecords files."""
    
    print("Calculate scaling factors...")
    scaling_factors = dict()
    
    if train_test_split != None:
        # only take training cells
        data_dict = {k: v for k, v in data_dict.items() if k in train_test_split["train"]}
    else:
        # only take non-secondary-test cells
        data_dict = {k: v for k, v in data_dict.items() if k.startswith('b3')}


    # Calculating max values for summary features
    for k in [cst.INTERNAL_RESISTANCE_NAME,
              cst.QD_NAME,
              cst.REMAINING_CYCLES_NAME,  # The feature "Current_cycles" will be scaled by the same scaling factor
              cst.DISCHARGE_TIME_NAME]:
        # Two max() calls are needed, one for every cell, one over all cells
        scaling_factors[k] = max([max(cell_v["summary"][k])
                                  for cell_k, cell_v in data_dict.items()
                                  for cycle_v in cell_v["cycles"].values()])
    
    # Calculating max values for detail features
    for k in [cst.QDLIN_NAME,
              cst.TDLIN_NAME]:
        # Two max() calls are needed, one over every cycle array, one over all cycles (all cells included)
        scaling_factors[k] = max([max(cycle_v[k])
                                  for cell_k, cell_v in data_dict.items()
                                  for cycle_v in cell_v["cycles"].values()])
    
    with open(csv_dir, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=scaling_factors.keys())
        writer.writeheader()  # Write the field names in the first line of the csv
        writer.writerow(scaling_factors)  # Write values to the corrent fields
    print("Saved scaling factors to {}".format(csv_dir))
    print("Scaling factors: {}".format(scaling_factors))
    return scaling_factors


def load_scaling_factors(csv_dir=cst.SCALING_FACTORS_DIR, gcloud_bucket=None):
    """Reads the scaling factors from a csv and returns them as a dict."""
    if gcloud_bucket:
        blob = gcloud_bucket.blob(csv_dir)
        names, values = blob.download_as_string().decode("utf-8").split("\r\n")[:2]  # Download and decode byte string.
        return {k: float(v) for k, v in zip(names.split(","), values.split(","))}
    else:
        with open(csv_dir, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                return {k: float(v) for k, v in row.items()}  # Return only the first found line with numeric values


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
