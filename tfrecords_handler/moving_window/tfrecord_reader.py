from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class TFRecordReader:

    def __init__(self, input_size, output_size, metadata_size):
        self.__input_size = input_size
        self.__output_size = output_size
        self.__metadata_size = metadata_size

    def train_data_parser(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.io.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32)
            })
        )
        return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["output"]

    def train_data_parser_2(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.io.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32)
            })
        )

        # noise = tf.random.normal(shape=tf.shape(sequence_parsed["input"]), mean=0.0, stddev=self.__gaussian_noise_stdev,
        #                          dtype=tf.float32)
        # training_input = sequence_parsed["input"] + noise

        return sequence_parsed["input"], sequence_parsed["output"]


    def validation_data_parser(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.io.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32),
                "metadata": tf.io.FixedLenSequenceFeature([self.__metadata_size], dtype=tf.float32)
            })
        )

        return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["output"], sequence_parsed[
            "metadata"]

    def validation_data_parser_2(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.io.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32),
                "metadata": tf.io.FixedLenSequenceFeature([self.__metadata_size], dtype=tf.float32)
            })
        )

        return sequence_parsed["input"]

    def validation_data_parser_3(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.io.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32),
                "metadata": tf.io.FixedLenSequenceFeature([self.__metadata_size], dtype=tf.float32)
            })
        )

        return context_parsed["sequence_length"]

    def validation_data_parser_4(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.io.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32),
                "metadata": tf.io.FixedLenSequenceFeature([self.__metadata_size], dtype=tf.float32)
            })
        )

        return sequence_parsed["metadata"]

    def validation_data_parser_5(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.io.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32),
                "metadata": tf.io.FixedLenSequenceFeature([self.__metadata_size], dtype=tf.float32)
            })
        )

        return sequence_parsed["output"]

    def validation_data_parser_6(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.io.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32),
                "metadata": tf.io.FixedLenSequenceFeature([self.__metadata_size], dtype=tf.float32)
            })
        )

        # noise = tf.random.normal(shape=tf.shape(sequence_parsed["input"]), mean=0.0, stddev=self.__gaussian_noise_stdev, dtype=tf.float32)
        # training_input = sequence_parsed["input"] + noise

        return sequence_parsed["input"], sequence_parsed["output"]

    def test_data_parser(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "metadata": tf.io.FixedLenSequenceFeature([self.__metadata_size], dtype=tf.float32)
            })
        )

        return sequence_parsed["input"]

    def test_data_parser_2(self, serialized_example):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.io.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.io.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "metadata": tf.io.FixedLenSequenceFeature([self.__metadata_size], dtype=tf.float32)
            })
        )

        return context_parsed["sequence_length"]