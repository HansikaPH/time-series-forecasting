from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter
import os

output_path = "../../../datasets/binary_data/kaggle_web_traffic/moving_window/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        input_size = 9,
        output_size = 59,
        train_file_path = '../../../datasets/text_data/kaggle_web_traffic/moving_window/kaggle_stl_59i9.txt',
        validate_file_path = '../../../datasets/text_data/kaggle_web_traffic/moving_window/kaggle_stl_59i9v.txt',
        test_file_path = '../../../datasets/text_data/kaggle_web_traffic/moving_window/kaggle_test_59i9.txt',
        binary_train_file_path = output_path + 'kaggle_stl_59i9.tfrecords',
        binary_validation_file_path = output_path + 'kaggle_stl_59i9v.tfrecords',
        binary_test_file_path = output_path + 'kaggle_test_59i9.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

    tfrecord_writer = TFRecordWriter(
        input_size=74,
        output_size=59,
        train_file_path='../../../datasets/text_data/kaggle_web_traffic/moving_window/kaggle_stl_59i74.txt',
        validate_file_path='../../../datasets/text_data/kaggle_web_traffic/moving_window/kaggle_stl_59i74v.txt',
        test_file_path='../../../datasets/text_data/kaggle_web_traffic/moving_window/kaggle_test_59i74.txt',
        binary_train_file_path=output_path + 'kaggle_stl_59i74.tfrecords',
        binary_validation_file_path=output_path + 'kaggle_stl_59i74v.tfrecords',
        binary_test_file_path=output_path + 'kaggle_test_59i74.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()