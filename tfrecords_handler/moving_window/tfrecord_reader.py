import tensorflow as tf

class TFRecordReader:

    def __init__(self, input_size, output_size):
        self.__input_size = input_size
        self.__output_size = output_size

    def train_data_parser(self, serialized_example):
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32)
            })
        )

        return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["output"]


    def validation_data_parser(self, serialized_example):
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32),
                "metadata": tf.FixedLenSequenceFeature([self.__output_size + 1], dtype=tf.float32)
            })
        )

        return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["output"], sequence_parsed[
            "metadata"]

    def validation_data_parser_without_stl(self, serialized_example):
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "output": tf.FixedLenSequenceFeature([self.__output_size], dtype=tf.float32),
                "metadata": tf.FixedLenSequenceFeature([1], dtype=tf.float32)
            })
        )

        return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["output"], sequence_parsed[
            "metadata"]

    def test_data_parser(self, serialized_example):
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "metadata": tf.FixedLenSequenceFeature([self.__output_size + 1], dtype=tf.float32)
            })
        )

        return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["metadata"]

    def test_data_parser_without_stl(self, serialized_example):
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.FixedLenSequenceFeature([self.__input_size], dtype=tf.float32),
                "metadata": tf.FixedLenSequenceFeature([1], dtype=tf.float32)
            })
        )

        return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["metadata"]