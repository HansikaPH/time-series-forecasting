#!/usr/bin/env bash

# with windowing

#stacking_smac_cocob
python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70.tfrecords --binary_valid_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70v.tfrecords --binary_test_file datasets/binary_data/NN5/moving_window/nn5_test_56i70.tfrecords --txt_test_file datasets/text_data/NN5/moving_window/nn5_test_56i70.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning smac --model_type stacking &

#stacking_smac_adagrad
python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70.tfrecords --binary_valid_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70v.tfrecords --binary_test_file datasets/binary_data/NN5/moving_window/nn5_test_56i70.tfrecords --txt_test_file datasets/text_data/NN5/moving_window/nn5_test_56i70.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning smac --model_type stacking &

#stacking_smac_adam
python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70.tfrecords --binary_valid_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70v.tfrecords --binary_test_file datasets/binary_data/NN5/moving_window/nn5_test_56i70.tfrecords --txt_test_file datasets/text_data/NN5/moving_window/nn5_test_56i70.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adam --hyperparameter_tuning smac --model_type stacking &

#stacking_bayesian_cocob
python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70.tfrecords --binary_valid_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70v.tfrecords --binary_test_file datasets/binary_data/NN5/moving_window/nn5_test_56i70.tfrecords --txt_test_file datasets/text_data/NN5/moving_window/nn5_test_56i70.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning bayesian --model_type stacking &

#stacking_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70.tfrecords --binary_valid_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70v.tfrecords --binary_test_file datasets/binary_data/NN5/moving_window/nn5_test_56i70.tfrecords --txt_test_file datasets/text_data/NN5/moving_window/nn5_test_56i70.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning bayesian --model_type stacking &

#stacking_bayesian_adam
python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70.tfrecords --binary_valid_file datasets/binary_data/NN5/moving_window/nn5_stl_56i70v.tfrecords --binary_test_file datasets/binary_data/NN5/moving_window/nn5_test_56i70.tfrecords --txt_test_file datasets/text_data/NN5/moving_window/nn5_test_56i70.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adam --hyperparameter_tuning bayesian --model_type stacking &

#attention_smac_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning smac --model_type attention &

#attention_smac_adagrad
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning smac --model_type attention &

#attention_bayesian_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning bayesian --model_type attention &

#attention_bayesian_adagrad
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --input_size 70 --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning bayesian --model_type attention &

# without windowing

#seq2seq_smac_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/non_moving_window/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/non_moving_window/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seq &

#seq2seq_smac_adagrad
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/non_moving_window/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/non_moving_window/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seq &

#seq2seq_smac_adam
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/non_moving_window/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/non_moving_window/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --forecast_horizon 56 --optimizer adam --hyperparameter_tuning smac --model_type seq2seq &

#seq2seq_bayesian_cocob
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/non_moving_window/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/non_moving_window/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --forecast_horizon 56 --optimizer cocob --hyperparameter_tuning bayesian --model_type seq2seq &

#seq2seq_bayesian_adagrad
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/non_moving_window/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/non_moving_window/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --forecast_horizon 56 --optimizer adagrad --hyperparameter_tuning bayesian --model_type seq2seq &

#seq2seq_bayesian_adam
#python ./generic_model_trainer.py --dataset_name nn5 --contain_zero_values 1 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn5 --binary_train_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56.tfrecords --binary_valid_file datasets/binary_data/NN5/non_moving_window/nn5_stl_56v.tfrecords --binary_test_file datasets/binary_data/NN5/non_moving_window/nn5_test_56.tfrecords --txt_test_file datasets/text_data/NN5/non_moving_window/nn5_test_56.txt --actual_results_file datasets/text_data/NN5/nn5_results.txt --forecast_horizon 56 --optimizer adam --hyperparameter_tuning bayesian --model_type seq2seq &

wait