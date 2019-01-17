from tfrecords_handler.non_moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        output_size = 59,
        train_file_path = '../../../datasets/text_data/kaggle_web_traffic/non_moving_window/kaggle_stl_59.txt',
        validate_file_path = '../../../datasets/text_data/kaggle_web_traffic/non_moving_window/kaggle_stl_59v.txt',
        test_file_path = '../../../datasets/text_data/kaggle_web_traffic/non_moving_window/kaggle_test_59.txt',
        binary_train_file_path = '../../../datasets/binary_data/kaggle_web_traffic/non_moving_window/kaggle_stl_59.tfrecords',
        binary_validation_file_path = '../../../datasets/binary_data/kaggle_web_traffic/non_moving_window/kaggle_stl_59v.tfrecords',
        binary_test_file_path = '../../../datasets/binary_data/kaggle_web_traffic/non_moving_window/kaggle_test_59.tfrecords',
        without_stl_decomposition=False
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()