from tfrecords_handler.non_moving_window.tfrecord_writer import TFRecordWriter
import os

output_path = "../../../datasets/binary_data/M4/non_moving_window/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    # macro data
    tfrecord_writer = TFRecordWriter(
        output_size = 18,
        train_file_path = '../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_macro_18.txt',
        validate_file_path = '../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_macro_18v.txt',
        test_file_path = '../../../datasets/text_data/M4/non_moving_window/m4_test_monthly_macro_18.txt',
        binary_train_file_path = output_path + 'm4_stl_monthly_macro_18.tfrecords',
        binary_validation_file_path = output_path + 'm4_stl_monthly_macro_18v.tfrecords',
        binary_test_file_path = output_path + 'm4_test_monthly_macro_18.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # micro data
    tfrecord_writer = TFRecordWriter(
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_micro_18.txt',
        validate_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_micro_18v.txt',
        test_file_path='../../../datasets/text_data/M4/non_moving_window/m4_test_monthly_micro_18.txt',
        binary_train_file_path=output_path + 'm4_stl_monthly_micro_18.tfrecords',
        binary_validation_file_path=output_path + 'm4_stl_monthly_micro_18v.tfrecords',
        binary_test_file_path=output_path + 'm4_test_monthly_micro_18.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # industry data
    tfrecord_writer = TFRecordWriter(
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_industry_18.txt',
        validate_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_industry_18v.txt',
        test_file_path='../../../datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt',
        binary_train_file_path=output_path + 'm4_stl_monthly_industry_18.tfrecords',
        binary_validation_file_path=output_path + 'm4_stl_monthly_industry_18v.tfrecords',
        binary_test_file_path=output_path + 'm4_test_monthly_industry_18.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # finance data
    tfrecord_writer = TFRecordWriter(
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_finance_18.txt',
        validate_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_finance_18v.txt',
        test_file_path='../../../datasets/text_data/M4/non_moving_window/m4_test_monthly_finance_18.txt',
        binary_train_file_path=output_path + 'm4_stl_monthly_finance_18.tfrecords',
        binary_validation_file_path=output_path + 'm4_stl_monthly_finance_18v.tfrecords',
        binary_test_file_path=output_path + 'm4_test_monthly_finance_18.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # other data
    tfrecord_writer = TFRecordWriter(
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_other_18.txt',
        validate_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_other_18v.txt',
        test_file_path='../../../datasets/text_data/M4/non_moving_window/m4_test_monthly_other_18.txt',
        binary_train_file_path=output_path + 'm4_stl_monthly_other_18.tfrecords',
        binary_validation_file_path=output_path + 'm4_stl_monthly_other_18v.tfrecords',
        binary_test_file_path=output_path + 'm4_test_monthly_other_18.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    # demographic data
    tfrecord_writer = TFRecordWriter(
        output_size=18,
        train_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_demo_18.txt',
        validate_file_path='../../../datasets/text_data/M4/non_moving_window/m4_stl_monthly_demo_18v.txt',
        test_file_path='../../../datasets/text_data/M4/non_moving_window/m4_test_monthly_demo_18.txt',
        binary_train_file_path=output_path + 'm4_stl_monthly_demo_18.tfrecords',
        binary_validation_file_path=output_path + 'm4_stl_monthly_demo_18v.tfrecords',
        binary_test_file_path=output_path + 'm4_test_monthly_demo_18.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()