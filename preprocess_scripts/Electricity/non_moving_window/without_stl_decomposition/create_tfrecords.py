from tfrecords_handler.non_moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':

    tfrecord_writer = TFRecordWriter(
        output_size = 12,
        train_file_path = '../../../../datasets/text_data/Electricity/non_moving_window/without_stl_decomposition/electricity_12.txt',
        validate_file_path = '../../../../datasets/text_data/Electricity/non_moving_window/without_stl_decomposition/electricity_12v.txt',
        test_file_path = '../../../../datasets/text_data/Electricity/non_moving_window/without_stl_decomposition/electricity_test_12.txt',
        binary_train_file_path = '../../../../datasets/binary_data/Electricity/non_moving_window/without_stl_decomposition/electricity_12.tfrecords',
        binary_validation_file_path = '../../../../datasets/binary_data/Electricity/non_moving_window/without_stl_decomposition/electricity_12v.tfrecords',
        binary_test_file_path = '../../../../datasets/binary_data/Electricity/non_moving_window/without_stl_decomposition/electricity_test_12.tfrecords',
        without_stl_decomposition=False
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()