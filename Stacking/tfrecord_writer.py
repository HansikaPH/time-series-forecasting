import tensorflow as tf
import numpy as np
import pandas as pd

train_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/stl_12i15.txt'
validate_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/stl_12i15v.txt'
test_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/cif12test.txt'

binary_train_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/Binary Files/stl_12i15.tfrecords'
binary_validation_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/Binary Files/stl_12i15v.tfrecords'
binary_test_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/Binary Files/cif12test.tfrecords'

# global lists for storing the data from files
list_of_trainig_inputs = []
list_of_training_labels = []
list_of_training_metadata = []
list_of_validation_inputs = []
list_of_validation_labels = []
list_of_validation_metadata = []
list_of_test_inputs = []
list_of_test_metadata = []

# Input/Output Window sizes
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# read the text data from text files
def read_text_data():
    # Reading the training dataset.
    train_df = pd.read_csv(train_file_path, nrows=10)

    float_cols = [c for c in train_df if train_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    train_df = pd.read_csv(train_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

    train_df = train_df.rename(columns={0: 'series'})

    # Returns unique number of time series in the dataset.
    series = np.unique(train_df['series'])

    # Construct input and output training tuples for each time series.
    for ser in series:
        oneSeries_df = train_df[train_df['series'] == ser]
        inputs_df = oneSeries_df.iloc[:, range(1, (INPUT_SIZE + 1))]
        labels_df = oneSeries_df.iloc[:, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
        metadata_df = oneSeries_df.iloc[:, range((INPUT_SIZE + OUTPUT_SIZE + 3), oneSeries_df.shape[1])]
        list_of_trainig_inputs.append(np.ascontiguousarray(inputs_df, dtype=np.float32))
        list_of_training_labels.append(np.ascontiguousarray(labels_df, dtype=np.float32))
        list_of_training_metadata.append(np.ascontiguousarray(metadata_df, dtype=np.float32))

    # Reading the validation dataset.
    val_df = pd.read_csv(validate_file_path, nrows=10)

    float_cols = [c for c in val_df if val_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    val_df = pd.read_csv(validate_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

    val_df = val_df.rename(columns={0: 'series'})
    val_df = val_df.rename(columns={(INPUT_SIZE + OUTPUT_SIZE + 3): 'level'})
    series = np.unique(val_df['series'])

    for ser in series:
        oneSeries_df = val_df[val_df['series'] == ser]
        inputs_df_test = oneSeries_df.iloc[:, range(1, (INPUT_SIZE + 1))]
        metadata_df = oneSeries_df.iloc[:, range((INPUT_SIZE + OUTPUT_SIZE + 3), oneSeries_df.shape[1])]
        labels_df_test = oneSeries_df.iloc[:, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
        list_of_validation_inputs.append(np.ascontiguousarray(inputs_df_test, dtype=np.float32))
        list_of_validation_labels.append(np.ascontiguousarray(labels_df_test, dtype=np.float32))
        list_of_validation_metadata.append(np.ascontiguousarray(metadata_df, dtype=np.float32))

    # Reading the test file.
    test_df = pd.read_csv(test_file_path, nrows=10)

    float_cols = [c for c in test_df if test_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    test_df = pd.read_csv(test_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

    test_df = test_df.rename(columns={0: 'series'})

    series1 = np.unique(test_df['series'])

    for ser in series1:
        test_series_df = test_df[test_df['series'] == ser]
        test_inputs_df = test_series_df.iloc[:, range(1, (INPUT_SIZE + 1))]
        metadata_df = test_series_df.iloc[:, range((INPUT_SIZE + 2), test_series_df.shape[1])]
        list_of_test_inputs.append(np.ascontiguousarray(test_inputs_df, dtype=np.float32))
        list_of_test_metadata.append(np.ascontiguousarray(metadata_df, dtype=np.float32))

# write the train and validation text data into tfrecord file
def write_to_tfrecord_file(tfrecord_file_path, list_of_inputs, list_of_metadata, list_of_outputs = []):

    writer = tf.python_io.TFRecordWriter(tfrecord_file_path)

    # write the training data file in tfrecords format
    for input, output, metadata in zip(list_of_inputs, list_of_outputs, list_of_metadata):

        sequence_length = input.shape[0]
        sequence_example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                "sequence_length" : tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_length]))
            }),
            feature_lists = tf.train.FeatureLists(feature_list={
                "input" : tf.train.FeatureList(feature=[
                    tf.train.Feature(float_list=tf.train.FloatList(value=input_sequence)) for input_sequence in input
                ]),
                "output" : tf.train.FeatureList(feature=[
                    tf.train.Feature(float_list=tf.train.FloatList(value=output_sequence)) for output_sequence in output
                ]),
                "metadata" : tf.train.FeatureList(feature=[
                    tf.train.Feature(float_list=tf.train.FloatList(value=metadata_sequence)) for metadata_sequence in metadata
                ])
            })
        )
        writer.write(sequence_example.SerializeToString())
    writer.close()

# write the test text data into tfrecord file
def write_test_data_to_tfrecord_file(tfrecord_file_path, list_of_inputs, list_of_metadata):

    writer = tf.python_io.TFRecordWriter(tfrecord_file_path)

    # write the training data file in tfrecords format
    for input, metadata in zip(list_of_inputs, list_of_metadata):

        sequence_length = input.shape[0]
        sequence_example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                "sequence_length" : tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_length]))
            }),
            feature_lists = tf.train.FeatureLists(feature_list={
                "input" : tf.train.FeatureList(feature=[
                    tf.train.Feature(float_list=tf.train.FloatList(value=input_sequence)) for input_sequence in input
                ]),
                "metadata" : tf.train.FeatureList(feature=[
                    tf.train.Feature(float_list=tf.train.FloatList(value=metadata_sequence)) for metadata_sequence in metadata
                ])
            })
        )
        writer.write(sequence_example.SerializeToString())
    writer.close()

read_text_data()
write_to_tfrecord_file(binary_train_file_path, list_of_trainig_inputs, list_of_training_metadata, list_of_training_labels)
write_to_tfrecord_file(binary_validation_file_path, list_of_validation_inputs, list_of_validation_metadata, list_of_validation_labels)
write_test_data_to_tfrecord_file(binary_test_file_path, list_of_test_inputs, list_of_test_metadata)