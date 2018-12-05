from tfrecords_handler.non_moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        output_size = 6,
        train_file_path = '../../../../datasets/text_data/CIF_2016/non_moving_window/without_stl_decomposition/cif_stl_6.txt',
        validate_file_path = '../../../../datasets/text_data/CIF_2016/non_moving_window/without_stl_decomposition/cif_stl_6v.txt',
        test_file_path = '../../../../datasets/text_data/CIF_2016/non_moving_window/without_stl_decomposition/cif_test_6.txt',
        binary_train_file_path = '../../../../datasets/binary_data/CIF_2016/non_moving_window/without_stl_decomposition/cif_stl_6.tfrecords',
        binary_validation_file_path = '../../../../datasets/binary_data/CIF_2016/non_moving_window/without_stl_decomposition/cif_stl_6v.tfrecords',
        binary_test_file_path = '../../../../datasets/binary_data/CIF_2016/non_moving_window/without_stl_decomposition/cif_test_6.tfrecords',
        without_stl_decomposition=True
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()