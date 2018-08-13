from tfrecords_handler.non_moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        output_size = 12,
        train_file_path = '../../../datasets/text_data/CIF_2016/non_moving_window/stl_12_seq2seq.txt',
        validate_file_path = '../../../datasets/text_data/CIF_2016/non_moving_window/stl_12_seq2seqv.txt',
        test_file_path = '../../../datasets/text_data/CIF_2016/non_moving_window/cif_12_seq2seq_test.txt',
        binary_train_file_path = '../../../datasets/binary_data/CIF_2016/non_moving_window/stl_12_seq2seq.tfrecords',
        binary_validation_file_path = '../../../datasets/binary_data/CIF_2016/non_moving_window/stl_12_seq2seqv.tfrecords',
        binary_test_file_path = '../../../datasets/binary_data/CIF_2016/non_moving_window/cif_12_seq2seq_test.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()