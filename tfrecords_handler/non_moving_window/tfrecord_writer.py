import tensorflow as tf
import numpy as np
import pandas as pd
import csv


class TFRecordWriter:

    def __init__(self, **kwargs):
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
        self.__list_of_validation_outputs = []
        self.__list_of_validation_metadata = []
        self.__list_of_test_inputs = []
        self.__list_of_test_metadata = []

        # Reading the training dataset.
        with open(self.__train_file_path) as train_file:
            train_data_reader = csv.reader(train_file, delimiter=" ")
            train_data_list = list(train_data_reader)

        for series in train_data_list:
            last_train_input_index = len(series) - self.__output_size - 2
            train_input_data = series[1: last_train_input_index + 1]
            train_output_data = series[(last_train_input_index + 2): len(series)]
            self.__list_of_training_inputs.append(np.ascontiguousarray(train_input_data, dtype=np.float32))
            self.__list_of_training_outputs.append(np.ascontiguousarray(train_output_data, dtype=np.float32))

        # Reading the validation dataset
        with open(self.__validate_file_path) as validate_file:
            validate_file_reader = csv.reader(validate_file, delimiter=" ")
            validate_data_list = list(validate_file_reader)

        for series in validate_data_list:
            meta_data_index = series.index("|#")
            output_index = series.index("|o")
            validate_input_data = series[1: output_index]
            validate_output_data = series[output_index + 1: meta_data_index]
            validate_meta_data = series[meta_data_index + 1:]
            self.__list_of_validation_inputs.append(np.ascontiguousarray(validate_input_data, dtype=np.float32))
            self.__list_of_validation_outputs.append(np.ascontiguousarray(validate_output_data, dtype=np.float32))
            self.__list_of_validation_metadata.append(np.ascontiguousarray(validate_meta_data, dtype=np.float32))

        # Reading the test file
        with open(self.__test_file_path) as test_file:
            test_file_reader = csv.reader(test_file, delimiter=" ")
            test_data_list = list(test_file_reader)

        for series in test_data_list:
            meta_data_index = series.index("|#")
            test_input_data = series[1: meta_data_index]
            test_meta_data = series[meta_data_index + 1:]
            self.__list_of_test_inputs.append(np.ascontiguousarray(test_input_data, dtype=np.float32))
            self.__list_of_test_metadata.append(np.ascontiguousarray(test_meta_data, dtype=np.float32))


    # write the train and validation text data into tfrecord file
    def write_train_data_to_tfrecord_file(self):
        writer = tf.python_io.TFRecordWriter(self.__binary_train_file_path,
                                             tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))

        # write the training data file in tfrecords format
        for input, output in zip(self.__list_of_training_inputs, self.__list_of_training_outputs):
            sequence_length = len(input)
            sequence_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    "sequence_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_length]))
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    "input": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=[input_data_element])) for input_data_element
                        in input
                    ]),
                    "output": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=[output_data_element])) for output_data_element
                        in output
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
        for input, output, metadata in zip(self.__list_of_validation_inputs, self.__list_of_validation_outputs,
                                           self.__list_of_validation_metadata):
            sequence_length = input.shape[0]
            sequence_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    "sequence_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_length]))
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    "input": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=[input_data_element])) for input_data_element
                        in input
                    ]),
                    "output": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=[output_data_element])) for output_data_element
                        in output
                    ]),
                    "metadata": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=[metadata_element])) for metadata_element in
                        metadata
                    ])
                })
            )
            writer.write(sequence_example.SerializeToString())
        writer.close()


    # write the test text data into tfrecord file
    def write_test_data_to_tfrecord_file(self):
        writer = tf.python_io.TFRecordWriter(self.__binary_test_file_path,
                                             tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))

        # write the training data file in tfrecords format
        for input, metadata in zip(self.__list_of_test_inputs, self.__list_of_test_metadata):
            sequence_length = input.shape[0]
            sequence_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    "sequence_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_length]))
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    "input": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=[input_data_element])) for input_data_element
                        in input
                    ]),
                    "metadata": tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=[metadata_element])) for metadata_element in
                        metadata
                    ])
                })
            )
            writer.write(sequence_example.SerializeToString())
        writer.close()
