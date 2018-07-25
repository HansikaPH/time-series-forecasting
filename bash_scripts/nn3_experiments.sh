#!/usr/bin/env bash

#stacking_smac_cocob
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type stacking

#stacking_smac_adagrad
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type stacking

#stacking_bayesian_cocob
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning bayesian --model_type stacking

#stacking_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning bayesian --model_type stacking

#seq2seq_smac_cocob
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seq

#seq2seq_smac_adagrad
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seq

#seq2seq_bayesian_cocob
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning bayesian --model_type seq2seq

#seq2seq_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning bayesian --model_type seq2seq

#attention_smac_cocob
#python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type attention

#attention_smac_adagrad
#python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type attention

#attention_bayesian_cocob
#python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning bayesian --model_type attention

#attention_bayesian_adagrad
#python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --binary_train_file datasets/NN3/binary_files/nn3_stl_18i13.tfrecords --binary_valid_file datasets/NN3/binary_files/nn3_stl_18i13v.tfrecords --binary_test_file datasets/NN3/binary_files/nn3_test_18i13.tfrecords --txt_test_file datasets/NN3/nn3_test_18i13.txt --actual_results_file datasets/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning bayesian --model_type attention