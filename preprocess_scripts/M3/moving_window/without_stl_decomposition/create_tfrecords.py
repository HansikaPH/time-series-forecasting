from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter

if __name__ == '__main__':
    # macro data
    tfrecord_writer = TFRecordWriter(
        input_size = 12,
        output_size = 18,
        train_file_path = '../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_macro_18i12.txt',
        validate_file_path = '../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_macro_18i12v.txt',
        test_file_path = '../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_macro_18i12.txt',
        binary_train_file_path = '../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_macro_18i12.tfrecords',
        binary_validation_file_path = '../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_macro_18i12v.tfrecords',
        binary_test_file_path = '../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_macro_18i12.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # micro data
    tfrecord_writer = TFRecordWriter(
        input_size=13,
        output_size=18,
        train_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_micro_18i13.txt',
        validate_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_micro_18i13v.txt',
        test_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_micro_18i13.txt',
        binary_train_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_micro_18i13.tfrecords',
        binary_validation_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_micro_18i13v.tfrecords',
        binary_test_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_micro_18i13.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # industry data
    tfrecord_writer = TFRecordWriter(
        input_size=13,
        output_size=18,
        train_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_industry_18i13.txt',
        validate_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_industry_18i13v.txt',
        test_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_industry_18i13.txt',
        binary_train_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_industry_18i13.tfrecords',
        binary_validation_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_industry_18i13v.tfrecords',
        binary_test_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_industry_18i13.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # finance data
    tfrecord_writer = TFRecordWriter(
        input_size=13,
        output_size=18,
        train_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_finance_18i13.txt',
        validate_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_finance_18i13v.txt',
        test_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_finance_18i13.txt',
        binary_train_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_finance_18i13.tfrecords',
        binary_validation_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_finance_18i13v.tfrecords',
        binary_test_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_finance_18i13.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # other data
    tfrecord_writer = TFRecordWriter(
        input_size=13,
        output_size=18,
        train_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_other_18i13.txt',
        validate_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_other_18i13v.txt',
        test_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_other_18i13.txt',
        binary_train_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_other_18i13.tfrecords',
        binary_validation_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_other_18i13v.tfrecords',
        binary_test_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_other_18i13.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # demographic data
    tfrecord_writer = TFRecordWriter(
        input_size=13,
        output_size=18,
        train_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_demo_18i13.txt',
        validate_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_monthly_demo_18i13v.txt',
        test_file_path='../../../../datasets/text_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_demo_18i13.txt',
        binary_train_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_demo_18i13.tfrecords',
        binary_validation_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_monthly_demo_18i13v.tfrecords',
        binary_test_file_path='../../../../datasets/binary_data/M3/moving_window/without_stl_decomposition/m3_test_monthly_demo_18i13.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()