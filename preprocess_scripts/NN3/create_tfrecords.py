from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        input_size = 13,
        output_size = 18,
        train_file_path = '../../datasets/text_data/NN3/moving_window/nn3_stl_18i13.txt',
        validate_file_path = '../../datasets/text_data/NN3/moving_window/nn3_stl_18i13v.txt',
        test_file_path = '../../datasets/text_data/NN3/moving_window/nn3_test_18i13.txt',
        binary_train_file_path = '../../datasets/binary_data/NN3/nn3_stl_18i13.tfrecords',
        binary_validation_file_path = '../../datasets/binary_data/NN3/nn3_stl_18i13v.tfrecords',
        binary_test_file_path = '../../datasets/binary_data/NN3/nn3_test_18i13.tfrecords',
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()