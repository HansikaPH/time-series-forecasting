import tensorflow as tf

class TFRecordReader:

    def train_data_parser(self, serialized_example):
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            context_features=({
                "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
            }),
            sequence_features=({
                "input": tf.FixedLenSequenceFeature([1], dtype=tf.float32),
                "output": tf.FixedLenSequenceFeature([1], dtype=tf.float32)
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
                "input": tf.FixedLenSequenceFeature([1], dtype=tf.float32),
                "output": tf.FixedLenSequenceFeature([1], dtype=tf.float32),
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
                "input": tf.FixedLenSequenceFeature([1], dtype=tf.float32),
                "metadata": tf.FixedLenSequenceFeature([1], dtype=tf.float32)
            })
        )

        return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["metadata"]