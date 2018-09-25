from tfrecords_handler.non_moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        output_size = 18,
        train_file_path = '../../../datasets/text_data/NN3/non_moving_window/nn3_stl_18.txt',
        validate_file_path = '../../../datasets/text_data/NN3/non_moving_window/nn3_stl_18v.txt',
        test_file_path = '../../../datasets/text_data/NN3/non_moving_window/nn3_test_18.txt',
        binary_train_file_path = '../../../datasets/binary_data/NN3/non_moving_window/nn3_stl_18.tfrecords',
        binary_validation_file_path = '../../../datasets/binary_data/NN3/non_moving_window/nn3_stl_18v.tfrecords',
        binary_test_file_path = '../../../datasets/binary_data/NN3/non_moving_window/nn3_test_18.tfrecords',
        without_stl_decomposition=False
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()