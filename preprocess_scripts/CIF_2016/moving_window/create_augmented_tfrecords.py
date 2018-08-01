from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        input_size = 15,
        output_size = 12,
        train_file_path = '../../datasets/CIF_2016/augmented_data/stl_12i15.txt',
        validate_file_path = '../../datasets/CIF_2016/augmented_data/stl_12i15v.txt',
        test_file_path = '../../datasets/CIF_2016/augmented_data/cif12test.txt',
        binary_train_file_path = '../../datasets/CIF_2016/binary_files/augmented_data/stl_12i15.tfrecords',
        binary_validation_file_path = '../../datasets/CIF_2016/binary_files/augmented_data/stl_12i15v.tfrecords',
        binary_test_file_path = '../../datasets/CIF_2016/binary_files/augmented_data/cif12test.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()