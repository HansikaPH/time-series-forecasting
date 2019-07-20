from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    # macro data
    tfrecord_writer = TFRecordWriter(
        input_size = 15,
        output_size = 18,
        train_file_path = '../../../datasets/text_data/M4/moving_window/m4_monthly_macro_18i15.txt',
        validate_file_path = '../../../datasets/text_data/M4/moving_window/m4_monthly_macro_18i15v.txt',
        test_file_path = '../../../datasets/text_data/M4/moving_window/m4_test_monthly_macro_18i15.txt',
        binary_train_file_path = '../../../datasets/binary_data/M4/moving_window/m4_monthly_macro_18i15.tfrecords',
        binary_validation_file_path = '../../../datasets/binary_data/M4/moving_window/m4_monthly_macro_18i15v.tfrecords',
        binary_test_file_path = '../../../datasets/binary_data/M4/moving_window/m4_test_monthly_macro_18i15.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # micro data
    tfrecord_writer = TFRecordWriter(
        input_size=15,
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_micro_18i15.txt',
        validate_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_micro_18i15v.txt',
        test_file_path='../../../datasets/text_data/M4/moving_window/m4_test_monthly_micro_18i15.txt',
        binary_train_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_micro_18i15.tfrecords',
        binary_validation_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_micro_18i15v.tfrecords',
        binary_test_file_path='../../../datasets/binary_data/M4/moving_window/m4_test_monthly_micro_18i15.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # industry data
    tfrecord_writer = TFRecordWriter(
        input_size=15,
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_industry_18i15.txt',
        validate_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_industry_18i15v.txt',
        test_file_path='../../../datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt',
        binary_train_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_industry_18i15.tfrecords',
        binary_validation_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_industry_18i15v.tfrecords',
        binary_test_file_path='../../../datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # finance data
    tfrecord_writer = TFRecordWriter(
        input_size=15,
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_finance_18i15.txt',
        validate_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_finance_18i15v.txt',
        test_file_path='../../../datasets/text_data/M4/moving_window/m4_test_monthly_finance_18i15.txt',
        binary_train_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_finance_18i15.tfrecords',
        binary_validation_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_finance_18i15v.tfrecords',
        binary_test_file_path='../../../datasets/binary_data/M4/moving_window/m4_test_monthly_finance_18i15.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # other data
    tfrecord_writer = TFRecordWriter(
        input_size=5,
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_other_18i5.txt',
        validate_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_other_18i5v.txt',
        test_file_path='../../../datasets/text_data/M4/moving_window/m4_test_monthly_other_18i5.txt',
        binary_train_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_other_18i5.tfrecords',
        binary_validation_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_other_18i5v.tfrecords',
        binary_test_file_path='../../../datasets/binary_data/M4/moving_window/m4_test_monthly_other_18i5.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # demographic data
    tfrecord_writer = TFRecordWriter(
        input_size=15,
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_demo_18i15.txt',
        validate_file_path='../../../datasets/text_data/M4/moving_window/m4_monthly_demo_18i15v.txt',
        test_file_path='../../../datasets/text_data/M4/moving_window/m4_test_monthly_demo_18i15.txt',
        binary_train_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_demo_18i15.tfrecords',
        binary_validation_file_path='../../../datasets/binary_data/M4/moving_window/m4_monthly_demo_18i15v.tfrecords',
        binary_test_file_path='../../../datasets/binary_data/M4/moving_window/m4_test_monthly_demo_18i15.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()