from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter
import os

output_path = "../../../../datasets/binary_data/CIF_2016/moving_window/without_stl_decomposition/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        input_size = 7,
        output_size = 6,
        train_file_path = '../../../../datasets/text_data/CIF_2016/moving_window/without_stl_decomposition/cif_6i7.txt',
        validate_file_path = '../../../../datasets/text_data/CIF_2016/moving_window/without_stl_decomposition/cif_6i7v.txt',
        test_file_path = '../../../../datasets/text_data/CIF_2016/moving_window/without_stl_decomposition/cif6test.txt',
        binary_train_file_path = output_path + 'cif_6i7.tfrecords',
        binary_validation_file_path = output_path + 'cif_6i7v.tfrecords',
        binary_test_file_path = output_path + 'cif6test.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()