from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        input_size = 74,
        output_size = 59,
        train_file_path = '../../../../datasets/text_data/kaggle_web_traffic/moving_window/without_stl_decomposition/kaggle_59i74.txt',
        validate_file_path = '../../../../datasets/text_data/kaggle_web_traffic/moving_window/without_stl_decomposition/kaggle_59i74v.txt',
        test_file_path = '../../../../datasets/text_data/kaggle_web_traffic/moving_window/without_stl_decomposition/kaggle_test_59i74.txt',
        binary_train_file_path = '../../../../datasets/binary_data/kaggle_web_traffic/moving_window/without_stl_decomposition/kaggle_59i74.tfrecords',
        binary_validation_file_path = '../../../../datasets/binary_data/kaggle_web_traffic/moving_window/without_stl_decomposition/kaggle_59i74v.tfrecords',
        binary_test_file_path = '../../../../datasets/binary_data/kaggle_web_traffic/moving_window/without_stl_decomposition/kaggle_test_59i74.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()