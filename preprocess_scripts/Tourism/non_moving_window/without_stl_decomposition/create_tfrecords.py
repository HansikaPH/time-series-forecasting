from tfrecords_handler.non_moving_window.tfrecord_writer import TFRecordWriter
import os

output_path = "../../../../datasets/binary_data/Tourism/non_moving_window/without_stl_decomposition/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        output_size = 24,
        train_file_path='../../../../datasets/text_data/Tourism/non_moving_window/without_stl_decomposition/tourism_24.txt',
        validate_file_path='../../../../datasets/text_data/Tourism/non_moving_window/without_stl_decomposition/tourism_24v.txt',
        test_file_path='../../../../datasets/text_data/Tourism/non_moving_window/without_stl_decomposition/tourism_test_24.txt',
        binary_train_file_path=output_path + 'tourism_24.tfrecords',
        binary_validation_file_path=output_path + 'tourism_24v.tfrecords',
        binary_test_file_path=output_path + 'tourism_test_24.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()