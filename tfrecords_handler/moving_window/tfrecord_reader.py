from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class TFRecordReader:

    def __init__(self, input_size, output_size, metadata_size):
        self.__input_size = input_size
        self.__output_size = output_size
        self.__metadata_size = metadata_size

    # training phase parsers
    def train_data_parser_for_training(self, serialized_example, gaussian_noise_stdev):
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
        noise =  tf.random.normal(shape=tf.shape(sequence_parsed["input"]), mean=0.0,
                                     stddev=gaussian_noise_stdev, dtype=tf.float32)
        input = sequence_parsed["input"] + noise
        return input, sequence_parsed["output"]

    def validation_data_input_parser(self, serialized_example):
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

    def validation_data_output_parser(self, serialized_example):
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

    def validation_data_lengths_parser(self, serialized_example):
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

    def validation_data_metadata_parser(self, serialized_example):
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

    # testing phase parsers
    def train_data_parser_for_testing(self, serialized_example, gaussian_noise_stdev):
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

        return sequence_parsed["input"], sequence_parsed["output"]

    def test_data_input_parser(self, serialized_example):
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

    def test_data_lengths_parser(self, serialized_example):
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