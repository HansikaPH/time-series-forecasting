from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        input_size = 70,
        output_size = 56,
        train_file_path = '../../../../datasets/text_data/NN5/moving_window/with_mean_level_normalization/nn5_stl_56i70.txt',
        validate_file_path = '../../../../datasets/text_data/NN5/moving_window/with_mean_level_normalization/nn5_stl_56i70v.txt',
        test_file_path = '../../../../datasets/text_data/NN5/moving_window/with_mean_level_normalization/nn5_test_56i70.txt',
        binary_train_file_path = '../../../../datasets/binary_data/NN5/moving_window/with_mean_level_normalization/nn5_stl_56i70.tfrecords',
        binary_validation_file_path = '../../../../datasets/binary_data/NN5/moving_window/with_mean_level_normalization/nn5_stl_56i70v.tfrecords',
        binary_test_file_path = '../../../../datasets/binary_data/NN5/moving_window/with_mean_level_normalization/nn5_test_56i70.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()