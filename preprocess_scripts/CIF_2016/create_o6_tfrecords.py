from tfrecords_handler.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        input_size = 7,
        output_size = 6,
        train_file_path = '../../datasets/CIF_2016/stl_6i7.txt',
        validate_file_path = '../../datasets/CIF_2016/stl_6i7v.txt',
        test_file_path = '../../datasets/CIF_2016/cif6test.txt',
        binary_train_file_path = '../../datasets/CIF_2016/binary_files/stl_6i7.tfrecords',
        binary_validation_file_path = '../../datasets/CIF_2016/binary_files/stl_6i7v.tfrecords',
        binary_test_file_path = '../../datasets/CIF_2016/binary_files/cif6test.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()