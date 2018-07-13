#!/usr/bin/env bash

#stacking_smac_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning smac --model_type stacking

#stacking_smac_adagrad
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning smac --model_type stacking

#stacking_bayesian_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning bayesian --model_type stacking

#stacking_bayesian_adagrad
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning bayesian --model_type stacking

#seq2seq_smac_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seq

#seq2seq_smac_adagrad
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seq

#seq2seq_bayesian_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning bayesian --model_type seq2seq

#seq2seq_bayesian_adagrad
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning bayesian --model_type seq2seq

#attention_smac_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning smac --model_type attention

#attention_smac_adagrad
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning smac --model_type attention

#attention_bayesian_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning bayesian --model_type attention

#attention_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --binary_train_file datasets/NN5/binary_files/nn5_stl_56i70.tfrecords --binary_valid_file datasets/NN5/binary_files/nn5_stl_56i70v.tfrecords --binary_test_file datasets/NN5/binary_files/nn5_test_56i70.tfrecords --txt_test_file datasets/NN5/nn5_test_56i70.txt --actual_results_file datasets/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning bayesian --model_type attention