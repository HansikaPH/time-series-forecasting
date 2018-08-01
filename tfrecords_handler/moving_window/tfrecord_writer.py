import tensorflow as tf
import numpy as np
import pandas as pd

class TFRecordWriter:

    def __init__(self, **kwargs):
        self.__input_size = kwargs['input_size']
        self.__output_size = kwargs['output_size']
        self.__train_file_path = kwargs['train_file_path']
        self.__validate_file_path = kwargs['validate_file_path']
        self.__test_file_path = kwargs['test_file_path']
        self.__binary_train_file_path = kwargs['binary_train_file_path']
        self.__binary_validation_file_path = kwargs['binary_validation_file_path']
        self.__binary_test_file_path = kwargs['binary_test_file_path']

    # read the text data from text files
    def read_text_data(self):
        self.__list_of_training_inputs = []
        self.__list_of_training_outputs = []
        self.__list_of_validation_inputs = []
        self.__list_of_validation_outputs =[]
        self.__list_of_validation_metadata = []
        self.__list_of_test_inputs = []
        self.__list_of_test_metadata = []

        # Reading the training dataset.
        train_df = pd.read_csv(self.__train_file_path, nrows=10)

        float_cols = [c for c in train_df if train_df[c].dtype == "float64"]
        float32_cols = {c: np.float32 for c in float_cols}

        train_df = pd.read_csv(self.__train_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

        train_df = train_df.rename(columns={0: 'series'})

        # Returns unique number of time series in the dataset.
        series = np.unique(train_df['series'])

        # Construct input and output training tuples for each time series.
        for ser in series:
            one_series_df = train_df[train_df['series'] == ser]
            inputs_df = one_series_df.iloc[:, range(1, (self.__input_size + 1))]
            outputs_df = one_series_df.iloc[:, range((self.__input_size + 2), (self.__input_size + self.__output_size + 2))]
            self.__list_of_training_inputs.append(np.ascontiguousarray(inputs_df, dtype=np.float32))
            self.__list_of_training_outputs.append(np.ascontiguousarray(outputs_df, dtype=np.float32))

        # Reading the validation dataset.
        val_df = pd.read_csv(self.__validate_file_path, nrows=10)

        float_cols = [c for c in val_df if val_df[c].dtype == "float64"]
        float32_cols = {c: np.float32 for c in float_cols}

        val_df = pd.read_csv(self.__validate_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

        val_df = val_df.rename(columns={0: 'series'})
        val_df = val_df.rename(columns={(self.__input_size + self.__output_size + 3): 'level'})
        series = np.unique(val_df['series'])

        for ser in series:
            one_series_df = val_df[val_df['series'] == ser]
            inputs_df_test = one_series_df.iloc[:, range(1, (self.__input_size + 1))]
            metadata_df = one_series_df.iloc[:, range((self.__input_size + self.__output_size + 3), one_series_df.shape[1])]
            outputs_df_test = one_series_df.iloc[:, range((self.__input_size + 2), (self.__input_size + self.__output_size + 2))]
            self.__list_of_validation_inputs.append(np.ascontiguousarray(inputs_df_test, dtype=np.float32))
            self.__list_of_validation_outputs.append(np.ascontiguousarray(outputs_df_test, dtype=np.float32))
            self.__list_of_validation_metadata.append(np.ascontiguousarray(metadata_df, dtype=np.float32))

        # Reading the test file.
        test_df = pd.read_csv(self.__test_file_path, nrows=10)

        float_cols = [c for c in test_df if test_df[c].dtype == "float64"]
        float32_cols = {c: np.float32 for c in float_cols}

        test_df = pd.read_csv(self.__test_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

        test_df = test_df.rename(columns={0: 'series'})

        series1 = np.unique(test_df['series'])

        for ser in series1:
            test_series_df = test_df[test_df['series'] == ser]
            test_inputs_df = test_series_df.iloc[:, range(1, (self.__input_size + 1))]
            metadata_df = test_series_df.iloc[:, range((self.__input_size + 2), test_series_df.shape[1])]
            self.__list_of_test_inputs.append(np.ascontiguousarray(test_inputs_df, dtype=np.float32))
            self.__list_of_test_metadata.append(np.ascontiguousarray(metadata_df, dtype=np.float32))

    # write the train and validation text data into tfrecord file
    def write_train_data_to_tfrecord_file(self):

        writer = tf.python_io.TFRecordWriter(self.__binary_train_file_path, tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))

        # write the training data file in tfrecords format
        for input, output in zip(self.__list_of_training_inputs, self.__list_of_training_outputs):

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
                    ])
                })
            )
            writer.write(sequence_example.SerializeToString())
        writer.close()

    # write the train and validation text data into tfrecord file
    def write_validation_data_to_tfrecord_file(self):

        writer = tf.python_io.TFRecordWriter(self.__binary_validation_file_path, tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.ZLIB))

        # write the training data file in tfrecords format
        for input, output, metadata in zip(self.__list_of_validation_inputs, self.__list_of_validation_outputs, self.__list_of_validation_metadata):
            sequence_length = input.shape[0]
            sequence_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    "sequence_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_length]))
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    "input": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=input_sequence)) for input_sequence in input
                    ]),
                    "output": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=output_sequence)) for output_sequence
                        in output
                    ]),
                    "metadata": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=metadata_sequence)) for metadata_sequence in metadata
                    ])
                })
            )
            writer.write(sequence_example.SerializeToString())
        writer.close()

    # write the test text data into tfrecord file
    def write_test_data_to_tfrecord_file(self):

        writer = tf.python_io.TFRecordWriter(self.__binary_test_file_path, tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))

        # write the training data file in tfrecords format
        for input, metadata in zip(self.__list_of_test_inputs, self.__list_of_test_metadata):

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