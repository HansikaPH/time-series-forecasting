#!/usr/bin/env bash

#stacking_smac_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o6.txt --input_size 7 --forecast_horizon 6 --optimizer cocob --hyperparameter_tuning smac --model_type stacking

#stacking_smac_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o6.txt --input_size 7 --forecast_horizon 6 --optimizer adagrad --hyperparameter_tuning smac --model_type stacking

#stacking_bayesian_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o6.txt --input_size 7 --forecast_horizon 6 --optimizer cocob --hyperparameter_tuning bayesian --model_type stacking

#stacking_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o6.txt --input_size 7 --forecast_horizon 6 --optimizer adagrad --hyperparameter_tuning bayesian --model_type stacking

#seq2seq_smac_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o6.txt --input_size 7 --forecast_horizon 6 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seq

#seq2seq_smac_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o6.txt --input_size 7 --forecast_horizon 6 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seq

#seq2seq_bayesian_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o6.txt --input_size 7 --forecast_horizon 6 --optimizer cocob --hyperparameter_tuning bayesian --model_type seq2seq

#seq2seq_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o6.txt --input_size 7 --forecast_horizon 6 --optimizer adagrad --hyperparameter_tuning bayesian --model_type seq2seq

#attention_smac_cocob
#python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o12.txt --input_size 7 --forecast_horizon 6 --optimizer cocob --hyperparameter_tuning smac --model_type attention

#attention_smac_adagrad
#python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o12.txt --input_size 7 --forecast_horizon 6 --optimizer adagrad --hyperparameter_tuning smac --model_type attention

#attention_bayesian_cocob
#python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o12.txt --input_size 7 --forecast_horizon 6 --optimizer cocob --hyperparameter_tuning bayesian --model_type attention

#attention_bayesian_adagrad
#python ./generic_model_trainer.py --dataset_name cif2016_O6 --contain_zero_values 0 --binary_train_file datasets/CIF_2016/binary_files/stl_6i7.tfrecords --binary_valid_file datasets/CIF_2016/binary_files/stl_6i7v.tfrecords --binary_test_file datasets/CIF_2016/binary_files/cif6test.tfrecords --txt_test_file datasets/CIF_2016/cif6test.txt --actual_results_file datasets/CIF_2016/cif_results_o12.txt --input_size 7 --forecast_horizon 6 --optimizer adagrad --hyperparameter_tuning bayesian --model_type attention